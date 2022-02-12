import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import root
import timm
import math


class Classifier(nn.Module):
    def __init__(self, cfg, load_weights=False, **kwargs):
        super(Classifier, self).__init__()
        # initialize the model
        self.model = timm.create_model(cfg.MODEL.NAME,
                                       pretrained=cfg.TRAIN.PRETRAINED,
                                       num_classes=cfg.DATASET.NUM_LABELS,
                                       drop_rate=cfg.MODEL.REGULARIZATION,
                                       drop_path_rate=cfg.MODEL.REGULARIZATION,
                                       **kwargs)

        if cfg.MODEL.NONLINEAR_HEAD:
            # use a non-linear classifier (with one hidden layer)
            if 'vit' in cfg.MODEL.NAME:
                self.model.head = nn.Sequential(nn.Linear(cfg.MODEL.NONLINEAR_HEAD_SIZE, cfg.MODEL.NONLINEAR_HEAD_SIZE),
                                                nn.Linear(cfg.MODEL.NONLINEAR_HEAD_SIZE, cfg.DATASET.NUM_LABELS))
            elif 'resnet' in cfg.MODEL.NAME:
                self.model.reset_classifier(cfg.MODEL.NONLINEAR_HEAD_SIZE)
                self.model.head.add_module('linear2', nn.Linear(cfg.MODEL.NONLINEAR_HEAD_SIZE, cfg.DATASET.NUM_LABELS))
            else:
                print(f'Model {cfg.MODEL.NAME} not supported for two linear layer, using one')

        # initialize loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # initialize forward function
        if cfg.MODEL.AVERAGE == 'logits':
            self.forward = self._forward_mean_logits
        elif cfg.MODEL.AVERAGE == 'embeddings':
            assert 'vit' in cfg.MODEL.NAME, 'mean embeddings only works with ViT models'
            self.forward = self._forward_mean_embeddings
        elif cfg.MODEL.AVERAGE == 'probabilities':
            self.forward = self._forward_mean_probabilities
            # using NLL loss because log_softmax is applied before averaging the study predictions
            self.loss_fn = nn.NLLLoss()
        elif cfg.MODEL.AVERAGE == 'max_probabilities':
            self.forward = self._forward_max_probabilities
        else:
            raise NotImplementedError

        # load weights
        self.path = root / cfg.MODEL.DIR / cfg.MODEL.FILE
        if load_weights:
            self.load_weights()

    def _forward_mean_probabilities(self, images, study_ids, *args, **kwargs):
        logits = self.model(images)
        p_y = F.log_softmax(logits, dim=-1)

        # get mean p_y for each study
        M = F.one_hot(study_ids).T.float()
        M = F.normalize(M, p=1, dim=1)
        p_y_mean = torch.mm(M, p_y)
        return p_y_mean

    def _forward_max_probabilities(self, images, study_ids, *args, **kwargs):
        logits = self.model(images)
        p_y = F.softmax(logits, dim=-1)

        # get logits of max p_y_pos for each study
        p_y_pos = p_y[:, 0]
        M = F.one_hot(study_ids).T.float()
        max_idx = (M * p_y_pos).argmax(dim=1)
        logits = logits[max_idx]

        return logits

    def _forward_mean_logits(self, images, study_ids, *args, **kwargs):
        logits = self.model(images)

        # get mean logits for each study
        M = F.one_hot(study_ids).T.float()
        M = F.normalize(M, p=1, dim=1)
        logits_mean = torch.mm(M, logits)

        return logits_mean

    def _forward_mean_embeddings(self, images, study_ids, *args, **kwargs):
        # get pooled embeddings
        embeddings = self.model.forward_features(images)

        # get mean embedding for each study
        M = F.one_hot(study_ids).T.float()
        M = F.normalize(M, p=1, dim=1)
        embeddings = torch.mm(M, embeddings)

        # get logits
        logits = self.model.head(embeddings)
        return logits

    def loss(self, batch):
        images = batch['images']
        labels = batch['labels']
        study_ids = batch['study_ids']
        extremities = batch['extremities']

        # get logits
        logits = self.forward(images, study_ids)
        p_y = F.softmax(logits, dim=-1)

        # compute loss
        loss = self.loss_fn(logits, labels)

        # get acc per extremity
        correct = (p_y.argmax(dim=-1) == labels).float()
        M = F.one_hot(extremities, num_classes=7).T.float()
        acc_per_extremity = torch.mm(M, correct.unsqueeze(-1)).squeeze(1) / M.sum(dim=1)

        outputs = {'loss': loss.item(),
                   'acc': correct.mean().item(),
                   'p_y_pos': p_y[:, 1].detach().cpu().numpy(),
                   'acc_per_extremity': acc_per_extremity.detach().cpu().numpy(),
                   }
        return loss, outputs

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load_weights(self):
        if self.path.is_file():
            self.load_state_dict(torch.load(self.path))
        else:
            print(f'Model weights not found at {self.path}')

    def predict(self, images):
        logits = self.model(images)
        y_hat = logits.argmax(dim=-1)
        return y_hat

    def get_attention_map(self, input):
        """
        Get attention map of last layer based on "Attention rollout" from
        Abnar & Zuidema (2012). Quantifying Attention Flow in Transformers. arXiv:2005.00928v2

        Input must be transformed, but not unsqueezed.

        This function is based on code from:
        https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
        """
        # patch input, and add cls token, add positional embedding
        transformer_input = self.model.patch_embed(input.unsqueeze(0))
        transformer_input = torch.cat((self.model.cls_token, transformer_input), dim=1)
        transformer_input = transformer_input + self.model.pos_embed

        # get attention maps of all layers
        attention_matrix = []
        for block in self.model.blocks:
            qkv = block.attn.qkv(transformer_input)[0].reshape(self.model.pos_embed.shape[1], 3, -1, 64)
            q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
            k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
            kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)
            attention_matrix.append(q @ kT)
            # get input for next block
            transformer_input = block(transformer_input)
        attention_matrix = torch.stack(attention_matrix, dim=0)

        # Average the attention weights across all heads.
        attention_matrix = torch.mean(attention_matrix, dim=1)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.
        residual_att = torch.eye(attention_matrix.size(1), device=attention_matrix.device)
        aug_att_mat = attention_matrix + residual_att
        aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1, keepdim=True)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size(), device=attention_matrix.device)
        joint_attentions[0] = aug_att_mat[0]
        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

        grid_size = int(math.sqrt(aug_att_mat.size(-1)))
        # get attention map of last layer from cls token
        att_map = joint_attentions[-1, 0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()

        return att_map


