class Config:
    def __init__(self, datalabel, recipes=[], **params):
        self.datalabel = datalabel
        self.workers = 8
        self.start_epoch = 0
        self.epochs = 30
        self.mask_strategy = 'min'
        self.mask_rate = 0.5

        self.lr = 1e-4
        self.scheduler_step = 5
        self.scheduler_gamma = 0.5
        self.optimizer = 'adamw'
        
        self.batch_size = 30
        self.resize=112
        self.a_dim = 8
        self.h_dim = 64
        self.embed_dim = 768
        
        self.dropout_rate = 0.0
        
        for i in recipes:
            self.recipe(i)

        for i in params:
            self.__setattr__(i, params[i])

        self.train_dataset = dict(datalabel=self.datalabel, resize=self.resize, augment=self.augment)
        self.val_dataset = dict(resize=self.resize, augment='augment_test')
    
    def recipe(self, name):
        if 'r3d' in name:
            self.batch_size=30
        if 'resnet' in name:
            self.batch_size=48
        if 'resnet18' in name:
            self.batch_size=2048
        if 'xception' in name:
            self.net='xception'
            self.batch_size=64
        if 'no-aug' in name:
            self.augment='augment_test'
        
        if 'ff' in name:
            self.augment='augment0'
        if 'celeb' in name:
            self.augment='augment0'
        if 'dfdc' in name:
            self.augment='augment0'
