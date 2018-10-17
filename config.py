import os, sys

path = os.path.split(__file__)[0]
# print("abs path is %s" %(os.path.abspath()))

config = {
    'batch_size' : 16,
    'total_epochs':60,
    'seed' : 1,

    'momentum': 0.9,
    'momentum2': 0.999,
    'weight_decay': 0.0004,

    'cuda' : True,
    'gpus' :1,
    'gpuargs' : {'num_workers': 4, 
               'pin_memory' : True
              },

    'model':'Resnet',

    'resume' : True,
    'optimizer' : 'Adam',
    'opt_config':{
        'lr' : 0.001,
        'betas' : (0.9, 0.99),
        'eps': 1e-8,
        'weight_decay': 0.004
    },
    'train_dataset':'MpiSintelClean',
    'validation_dataset':'MpiSintel_Test',

    'save' :'%s/work/' % path,
    'gradient_clip': False,

    'image_folder_train' : {
        'root' : '%s/' % path,
        # 'dstype' : 'all_list.txt',
        'dstype' : 'sintal_trainset.txt',
        'crop_size': [256, 256],
        'render_size': [512, 1024],
        'replicates': 1,
        'train':True
    },
    'image_folder_val' : {
        'root' : '%s/' % path,
        # 'dstype' : 'all_list.txt',
        'dstype' : 'sintal_testset.txt',
        'crop_size': [256, 256],
        'render_size': [384, 1024],
        'replicates': 1,
        'train':False
    },

    'skip_validate': False,
    'skip_training': False,

    'inference':False,
    'validate_frequency':1,#frequency for the number of epotch
    'log_frequency': 1, #frequency for the number of epotch
    'inference_data_root':'',
    'train':True,
    'train_data_root':'',
    'validation':True,
    'validation_data_root':'',
    'save_iamge':'%s/work/img/' % path
}

# print(config['save'])