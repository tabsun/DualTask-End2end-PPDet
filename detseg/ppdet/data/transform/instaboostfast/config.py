import numpy as np
        
class InstaBoostConfig:
    def __init__(self, action_prob: float = 0., 
                 scale: tuple = (0.8, 1.2), 
                 dx: float = 15, dy: float = 15,
                 theta=(-1, 1), color_prob=0.5, 
                 sync_nums_with_pillar=[0,1,2], sync_prob_with_pillar=[0.3, 0.4, 0.3], sync_heatmap_flag_with_pillar=False,
                 sync_nums_only_obj=[0,1,2], sync_prob_only_obj=[0.3, 0.4, 0.3], sync_heatmap_flag_only_obj=False):
        """
        :param action_prob: tuple of corresponding action probabilities. Should be the same length as action_candidate
        :param scale: tuple of (min scale, max scale)
        :param dx: the maximum x-axis shift will be  (instance width) / dx
        :param dy: the maximum y-axis shift will be  (instance height) / dy
        :param theta: tuple of (min rotation degree, max rotation degree)
        :param color_prob: the probability of images for color augmentation
        :param heatmap_flag: whether to use heatmap guided
        """
        assert len(sync_nums_with_pillar) == len(sync_prob_with_pillar), 'sync_nums_with_pillar & prob length mismatch'
        assert (action_prob >= 0 and action_prob <= 1.), 'action probability must >= 0. and <= 1.'
        assert len(scale) == 2, 'scale should have 2 items (min scale, max scale)'
        assert len(theta) == 2, 'theta should have 2 items (min theta, max theta)'

        self.action_prob = action_prob
        self.scale = scale
        self.dx = dx
        self.dy = dy
        self.theta = theta
        self.color_prob = color_prob

        self.sync_nums_with_pillar = sync_nums_with_pillar
        self.sync_prob_with_pillar = sync_prob_with_pillar
        self.sync_heatmap_flag_with_pillar = sync_heatmap_flag_with_pillar

        self.sync_nums_only_obj = sync_nums_only_obj
        self.sync_prob_only_obj = sync_prob_only_obj
        self.sync_heatmap_flag_only_obj = sync_heatmap_flag_only_obj

    def get_dict(self):
        return {
                'action_prob': self.action_prob,
                'scale': self.scale,
                'dx': self.dx,
                'dy': self.dy,
                'theta': self.theta,
                'color_prob': self.color_prob,

                'sync_nums_with_pillar': self.sync_nums_with_pillar,
                'sync_prob_with_pillar': self.sync_prob_with_pillar,
                'sync_heatmap_flag_with_pillar': self.sync_heatmap_flag_with_pillar,

                'sync_nums_only_obj': self.sync_nums_only_obj,
                'sync_prob_only_obj': self.sync_prob_only_obj,
                'sync_heatmap_flag_only_obj': self.sync_heatmap_flag_only_obj 
                }
