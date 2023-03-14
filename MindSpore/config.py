class Config:
    def __init__(self, datalabel, recipes=[], **params):
        self.datalabel = datalabel
        self.workers = 8
        self.start_epoch = 0
        self.epochs = 100
        self.max_epoch = 196
        self.mask_strategy = 'min'
        self.mask_rate = 0.5

        self.lr = 2e-4
        self.weight_decay = 1e-6
        self.bn_weight_decay = 0.0
        self.scheduler_step = 5
        self.scheduler_gamma = 0.5
        self.momentum = 0.9
        self.adam_betas = (0.9, 0.999)
        self.optimizer = 'adamw'

        self.lr_policy = 'cosine'
        self.warmup_epochs = 30
        self.warmup_lr = 1e-6
        
        self.batch_size = 16
        self.resize=112
        self.a_dim = 8
        self.h_dim = 64
        self.embed_dim = 768
        
        self.dropout_rate = 0.1
        
        for i in recipes:
            self.recipe(i)

        for i in params:
            self.__setattr__(i, params[i])

        self.train_dataset = dict(datalabel=self.datalabel, resize=self.resize, augment=self.augment)
        self.val_dataset = dict(resize=self.resize, augment='augment_test')
    
    def recipe(self, name):
        if 'vit' in name:
            self.batch_size=8
        if 'ff' in name:
            if 'ff-5' in name:
                self.num_classes=5
            self.augment='augment0'
        if 'celeb' in name:
            self.augment='augment0'
        if 'dfdc' in name:
            self.augment='augment0'
        if 'uadfv' in name:
            self.augment='augment0'
        if 'xception' in name:
            self.net='xception'
            self.batch_size=64

        if 'efficient' in name:
            self.net=name
            self.batch_size=10
            scale=int(name.split('b')[-1])
            sizes=[224, 240, 260, 300, 380, 456, 528, 600, 672]
            self.resize=sizes[scale]
        if 'r3d' in name:
            self.batch_size=40
        if 'resnet' in name:
            self.batch_size=48
        if 'addnet' in name:
            self.batch_size=60
        if 'resnet18' in name:
            self.batch_size=2048
        if 'no-aug' in name:
            self.augment='augment_test'
