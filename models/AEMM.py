import torch.nn as nn
import os
from datasets import ContinualDataset
from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser
import torch
import numpy as np
import torch.nn.functional as F
import copy
from models.utils.feature_distance import FeatureDistance


class AEMM(ContinualModel):
    NAME = 'AEMM'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--alpha', type=float, required=True,
                            help='KL weight.',default=0.3)
        parser.add_argument('--beta', type=float, required=True,
                            help='mmd/mse weight.')
        parser.add_argument('--metric', type=str, default='mse')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(AEMM, self).__init__(backbone, loss, args, transform, dataset=dataset)
        self.currentTaskIndex = 0
        self.teacher_model = None 

    def end_task(self, dataset) -> None:
        self.teacher_model = copy.deepcopy(self.net)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        for param in self.net.classifierArr[self.currentTaskIndex].parameters():
            param.requires_grad = False
        

    def begin_task(self, dataset) -> None:
        n = np.shape(self.net.classifierArr)[0]
        self.net.createNewExpert(dataset.N_CLASSES)
        self.currentTaskIndex = n
        self.net.currentTaskIndex = self.currentTaskIndex

        self.opt = self.get_optimizer()

    def forward(self, x: torch.Tensor, k: int) -> torch.Tensor:
        self.net.currentTaskIndex = k
        return self.net(x)
    
    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        outputs, evolved_feats = self.net(inputs, 'both')
        loss = self.loss(outputs, labels)

        if self.current_task > 0 and (self.args.alpha > 0.0 or self.args.beta > 0.0): 
            with torch.no_grad():
                teacher_evolved_feats = self.teacher_model(inputs, 'feature')

            if self.args.alpha > 0.0:
                stu_inv_feature, stu_evo_feature = self.net(inputs, 'features')
                tea_inv_feature, tea_evo_feature = self.teacher_model(inputs, 'features')
                kl_loss = 0.0
                for i, expert in enumerate(self.net.classifierArr):
                    tea_output = expert(tea_inv_feature, tea_evo_feature)
                    stu_output = expert(stu_inv_feature, stu_evo_feature)
                    kl_loss += kl_loss_fun(stu_output, tea_output)
                loss += self.args.alpha * kl_loss / len(self.net.classifierArr)

            if self.args.beta > 0.0:
                total_feat_loss = sum(FeatureDistance.compute_distance(f, t, metric=self.args.metric)
                      for f, t in zip(evolved_feats, teacher_evolved_feats))
                feat_loss = total_feat_loss / len(evolved_feats) if evolved_feats else 0.0
                loss += self.args.beta * feat_loss

        loss.backward()
        total_loss = loss.item()
        self.net.currentTaskIndex = self.currentTaskIndex
        self.opt.step()
        
        return total_loss

def kl_loss_fun(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=1)

    student_prob = (student_feat + 1) / 2
    teacher_prob = (teacher_feat.detach() + 1) / 2

    loss_kld = F.kl_div(
        torch.log(student_prob + 1e-10),
        teacher_prob,
        reduction='batchmean'
    )
    return loss_kld
