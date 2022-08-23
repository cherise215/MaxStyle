import torchsample.transforms as ts
from src.dataset_loader._utils.affine_transform import MyRandomFlip, MySpecialCrop, MyPad, MyRandomChoiceRotate,MyRandomAffine
from src.dataset_loader._utils.intensity_transform import RandomGamma, MyNormalizeMedicPercentile, MyRandomPurtarbation, MyRandomPurtarbationV2, RandomBrightnessFluctuation
from src.dataset_loader._utils.elastic_transform import MyElasticTransform, MyElasticTransformCoarseGrid

class Transformations:
    def __init__(self, data_aug_policy_name, pad_size=(80, 80, 1), crop_size=(80, 80, 1)):
        self.name = data_aug_policy_name
        self.pad_size = pad_size
        self.crop_size = crop_size

    def get_transformation(self):
        # replicate experiment settings (traditional methods: affine,elastic, gamma) from 'task-driven data
        # augmentation' https://arxiv.org/pdf/1902.05396.pdf
        aug_config = {
            'no_aug': self.no_aug,
            'gamma': self.gamma_aug,
            'gamma_scale': self.gamma_scale,
            'affine': self.affine_aug,
            'scale': self.scale_aug,
            'elastic': self.elastic_aug,
            'elastic_scale': self.elastic_scale,
            'gamma_elastic': self.gamma_elastic_aug,
            'affine_elastic': self.affine_elastic_aug,
            'affine_gamma': self.affine_elastic_aug,
            'affine_gamma_elastic': self.affine_gamma_elastic_aug,
            'ACDC_affine': self.ACDC_affine_aug,
            'ACDC_affine_perturb': self.ACDC_affine_perturb_aug,
            'ACDC_affine_perturb_v2': self.ACDC_affine_perturb_v2,
            'ACDC_affine_elastic': self.ACDC_affine_elastic_aug,
            'ACDC_affine_intensity': self.ACDC_affine_intensity_aug,
            'ACDC_affine_elastic_intensity': self.ACDC_affine_elastic_intensity_aug,
            'ACDC_affine_elastic_intensity_v2': self.ACDC_affine_elastic_intensity_aug_v2,
            'ACDC_affine_elastic_bias': self.ACDC_affine_bias_elastic,
            'ACDC_affine_all': self.ACDC_affine_bias_elastic_intensity,
            'Atrial_basic': self.Atrial_basic,
            'Atrial_perturb': self.Atrial_Perturb,
            'Prostate_affine_elastic_intensity': self.Prostate_affine_elastic_intensity_aug,
            'elastic_v2': self.elasticv2_aug,
            'UKBB_affine_elastic_intensity_aug': self.UKBB_affine_elastic_intensity_aug,

        }[self.name]()

        return self.get_transform(aug_config)

    def get_transform(self, config):
        train_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                      ts.ToTensor(),
                                      ts.ChannelsFirst(),
                                      ts.TypeCast(['float', 'float']),
                                      # geometric transformation
                                      MyRandomAffine(rotation_range=config['rotate_val'],
                                                translation_range=config['shift_val'],
                                                shear_range=config['shear_val'],
                                                zoom_range=config['scale_val'], interp=('bilinear', 'nearest')),
                                      MyElasticTransform(is_labelmap=[False, True], p_thresh=config['elastic_prob']),
                                      MyElasticTransformCoarseGrid(
                                        is_labelmap=[False, True], p_thresh=config['elastic_probv2']),
                                      ts.RandomCrop(size=self.crop_size),

                                      MyRandomFlip(h=config['flip_flag'][0], v=config['flip_flag'][1],
                                                   p=config['flip_flag'][2]),
                                      MyRandomChoiceRotate(values=config['rotate_groups'],
                                                           interp=['bilinear', 'nearest']),

                                      # intensity transformation
                                      MyRandomPurtarbation(p=config['perturb_prob'], max_sigma=config['max_sigma'],
                                                           flag=[True, False],
                                                           multi_control_points=config['multi_control_points'],
                                                           add_noise=config['add_noise'],
                                                           epsilon=config['noise_epsilon']),

                                      MyRandomPurtarbationV2(p=config['perturb_v2_prob'], magnitude=config['perturb_v2_bias_magnitude'],
                                                             flag=[True, False],
                                                             ms_control_point_spacing=config['ms_control_point_spacing'],
                                                             add_noise=config['perturb_v2_add_noise'],
                                                             epsilon=config['perturb_v2_noise_epsilon'], debug=False),

                                      RandomBrightnessFluctuation(p=config['intensity_prob'], flag=[True, False]),
                                                                        
                                      MyNormalizeMedicPercentile(norm_flag=(True, False), min_val=0.0, max_val=1.0,
                                                                 perc_threshold=(0.0, 100.0)),
                                     ts.TypeCast(['float', 'long'])
                                    ])

        norm_transform = ts.Compose([
                                        ts.PadNumpy(size=self.pad_size),
                                        ts.ToTensor(),
                                        ts.ChannelsFirst(),
                                        MySpecialCrop(size=self.crop_size, crop_type=0),
                                        MyNormalizeMedicPercentile(norm_flag=(True, False), min_val=0.0, max_val=1.0,
                                                                 perc_threshold=(0.0, 100.0)),
                                        ts.TypeCast(['float', 'long'])
                                    ])

        # test_transform only support image as input
        test_transform = ts.Compose([ts.PadNumpy(size=self.pad_size),
                                     ts.ToTensor(),
                                     ts.ChannelsFirst(),
                                     ts.TypeCast(['float']),
                                     MySpecialCrop(size=self.crop_size, crop_type=0),
                                    #  MyNormalizeMedicPercentile(norm_flag=(True), min_val=0.0, max_val=1.0,
                                    #                             perc_threshold=(0.0, 100.0)),
                                     ts.TypeCast(['float'])

                                     ])

        return {'train': {'aug': train_transform,'norm':norm_transform}, 
                'validate':{'aug':norm_transform,'norm':norm_transform}, 
                'test': {'aug':test_transform,'norm':test_transform},
                'aug_validate': {'aug':train_transform,'norm':norm_transform}}

    def no_aug(self):
        config = {
            # affine augmentation
            'flip_flag': [False, False, 0.0],
            'shift_val': (0., 0.),
            'rotate_val': 0,
            'scale_val': (1., 1.),
            'rotate_groups': [],
            # contrast constrast aug
            'intensity_prob': 0,  # contrast and brightness
            'gamma_prob': 0.,
            # 'gamma_range': [0.8, 1.2],
            # deformation aug
            'elastic_prob': 0.,
            'shear_val': 0,
            'elastic_probv2': 0,
            # perturbation v1
            'perturb_prob': 0.,
            'max_sigma': 16,
            'multi_control_points': [4],
            'add_noise': False,
            'noise_epsilon': 0.01,

            # perturbation v2:
            'perturb_v2_prob': 0.,
            'perturb_v2_bias_magnitude': 0.2,
            'ms_control_point_spacing': [32],
            'perturb_v2_add_noise': False,
            'perturb_v2_noise_epsilon': 0.01
        }
        return config

    def scale_aug(self):
        config = self.no_aug()
        config['scale_val'] = (0.8, 1.2)
        return config

    def affine_aug(self):
        config = self.no_aug()
        # config['flip_flag'] = [False, True, 0.5]
        config['shift_val'] = (0.1, 0.1)
        config['rotate_val'] = 15
        config['scale_val'] = (0.9, 1.1)

        return config

    def Atrial_basic(self):
        config = self.no_aug()
        config['flip_flag'] = [True, True, 0.5]
        config['shift_val'] = (0.1, 0.1)
        config['rotate_val'] = 10
        config['scale_val'] = (0.7, 1.3)
        # config['gamma_range'] = (0.8, 2.0)
        # config['gamma_prob'] = 0.5

        return config

    def Prostate_affine_elastic_intensity_aug(self):
        config = self.no_aug()
        config['flip_flag'] = [True, True, 0.5]
        config['shift_val'] = (0.1, 0.1)
        config['rotate_val'] = 15
        config['scale_val'] = (0.8, 1.2)
        config['intensity_prob'] = 0.5
        config['elastic_prob'] = 0.5
        return config

    def Atrial_Perturb(self):
        config = self.no_aug()
        config['flip_flag'] = [True, True, 0.5]
        config['shift_val'] = (0.1, 0.1)
        config['rotate_val'] = 10
        config['scale_val'] = (0.7, 1.3)
        # config['gamma_range'] = (0.8, 2.0)
        # config['gamma_prob'] = 0.5
        config['perturb_prob'] = 0.5
        config['max_sigma'] = 16
        config['multi_control_points'] = [2, 4, 8]
        return config

    def ACDC_affine_aug(self):
        config = self.no_aug()
        config['flip_flag'] = [True, True, 0.2]
        config['shift_val'] = (0., 0.)
        config['rotate_val'] = 15
        config['scale_val'] = (0.8, 1.1)
        config['rotate_groups'] = [45 * i for i in range(8)]
        return config

    def ACDC_affine_intensity_aug(self):
        config = self.ACDC_affine_aug()
        config['intensity_prob'] = 0.5
        return config

    def ACDC_affine_elastic_intensity_aug(self):
        config = self.ACDC_affine_aug()
        config['intensity_prob'] = 0.5
        config['elastic_prob'] = 0.5

        return config

    def UKBB_affine_elastic_intensity_aug(self):
        config = self.ACDC_affine_aug()
        config['intensity_prob'] = 0.5
        config['elastic_prob'] = 0.5
        config['scale_val'] = [0.7, 1.4]
        config['rotate_val'] = 30
        config['flip_flag'] = [True, True, 0.5]
        config['shift_val'] = (0., 0.) ## use random crop already.
        return config

    def ACDC_affine_elastic_intensity_aug_v2(self):
        config = self.ACDC_affine_aug()
        config['intensity_prob'] = 0.5
        config['elastic_probv2'] = 0.5

        return config

    def ACDC_affine_perturb_aug(self):
        config = self.ACDC_affine_aug()
        config['perturb_prob'] = 0.5
        config['max_sigma'] = 16
        config['multi_control_points'] = [2, 4, 8]
        config['add_noise'] = True
        config['epsilon'] = 0.01
        return config

    def ACDC_affine_perturb_v2(self):
        config = self.ACDC_affine_aug()
        config['perturb_v2_prob'] = 0.5
        config['perturb_v2_bias_magnitude'] = 0.3
        config['ms_control_point_spacing'] = [64, 1]
        config['perturb_v2_add_noise'] = True
        config['perturb_v2_noise_epsilon'] = 0.01
        return config

    def ACDC_affine_bias_elastic(self):
        config = self.ACDC_affine_aug()
        config['perturb_v2_prob'] = 0.5
        config['perturb_v2_bias_magnitude'] = 0.3
        config['ms_control_point_spacing'] = [64, 1]
        config['perturb_v2_add_noise'] = True
        config['perturb_v2_noise_epsilon'] = 0.01
        config['elastic_prob'] = 0.5
        return config

    def ACDC_affine_bias_elastic_intensity(self):
        config = self.ACDC_affine_bias_elastic()
        config['intensity_prob'] = 0.5

        return config

    def ACDC_affine_elastic_aug(self):
        config = self.ACDC_affine_aug()
        config['elastic_prob'] = 0.5
        return config

    def affine_elastic_aug(self):
        config = self.affine_aug()
        config['elastic_prob'] = 0.5
        return config

    def affine_gamma_aug(self):
        config = self.affine_aug()
        config['gamma_prob'] = 0.5
        return config

    def affine_gamma_elastic_aug(self):
        print ('gamma is not implemented')
        config = self.affine_aug()
        config['gamma_prob'] = 0.5
        config['elastic_prob'] = 0.5
        # config['gamma_range'] = [0.8, 1.2]

        return config

    def gamma_aug(self):
        # config = self.no_aug()
        # config['gamma_prob'] = 0.5
        # config['gamma_range'] = [0.8, 1.2]
        raise NotImple

        return config

    def elastic_aug(self):
        config = self.no_aug()
        config['elastic_prob'] = 1
        return config

    def elasticv2_aug(self):
        config = self.no_aug()
        config['elastic_probv2'] = 1
        return config

    def gamma_scale(self):
        config = self.no_aug()
        config['gamma_prob'] = 0.5
        config['gamma_range'] = [0.8, 1.2]
        config['scale_val'] = [0.9, 1.1]
        return config

    def elastic_scale(self):
        config = self.no_aug()
        config['elastic_prob'] = 0.5
        config['scale_val'] = [0.9, 1.1]
        return config

    def gamma_elastic_aug(self):
        config = self.no_aug()
        config['gamma_prob'] = 0.5
        config['gamma_range'] = [0.8, 1.2]
        config['elastic_prob'] = 0.5
        return config
