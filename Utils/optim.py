import torch

def optimization_settings(parameters, config=None, learning_rate=1e-3, optim_type='SGD', sched_type='None', num_epochs=240):
    '''
        Optimizer options: "SGD", "Adam", "AdamW"\n
        Scheduler options: "None", "Step", "Cosine", "Cyclic1", "Cyclic2", "Cyclic3", "MultiStep", "OneCycle"
    '''
    if config!=None:
        learning_rate, optim_type, sched_type = config.learning_rate, config.optim_type, config.scheduler
        num_epochs = config.n_epochs
        if isinstance(learning_rate, str): learning_rate = float(learning_rate)
        # if isinstance(num_epochs, str): num_epochs = int(num_epochs)

    optimizer_options = ['SGD', 'Adam', 'AdamW']
    scheduler_options = ['None', 'Step', 'Cosine', 'Cyclic1', 'Cyclic2', 'Cyclic3', 'MultiStep','OneCycle']
    print(scheduler_options)
    print(sched_type)
    if optim_type not in optimizer_options: raise TypeError('Wrong optimizer name.')
    if sched_type not in scheduler_options: raise TypeError(f'Wrong scheduler name: {sched_type}')

    if optim_type==optimizer_options[0]:
        optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        is_momentum = True
    elif optim_type==optimizer_options[1]:
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        is_momentum = False
    elif optim_type==optimizer_options[2]:
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
        is_momentum = False
    
    if sched_type==scheduler_options[0]:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5000, gamma=1.0)
    elif sched_type==scheduler_options[1]:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.75)
    elif sched_type==scheduler_options[2]:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=learning_rate/1000)
    elif sched_type==scheduler_options[3]:
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate/100, max_lr=learning_rate, step_size_up=10, step_size_down=20, mode='triangular', cycle_momentum=is_momentum)
    elif sched_type==scheduler_options[4]:
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate/100, max_lr=learning_rate, step_size_up=5, step_size_down=15, mode='triangular2', cycle_momentum=is_momentum)
    elif sched_type==scheduler_options[5]:
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate/100, max_lr=learning_rate, step_size_up=5, step_size_down=15, mode='exp_range', cycle_momentum=is_momentum, gamma=0.98)
    elif sched_type==scheduler_options[6]:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160,200,240,270,300], gamma=0.75)
    elif sched_type==scheduler_options[7]:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=num_epochs, pct_start=0.2, cycle_momentum=is_momentum, div_factor=10.0, final_div_factor=1e3, three_phase=False)
    
    return optimizer, lr_scheduler