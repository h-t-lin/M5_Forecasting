import warnings
from Utils.Assistant_func import SnL

class Config():
    def __init__(self, args=None, paras:dict ={}) -> None:
        if args is not None: self.set_paras(self._extractNone_(vars(args)))
        self.set_paras(paras)
    
    def set_paras(self, paras={}, replace=True):
        '''If replace is true, the existing attributes will be overwritten if needed.'''
        if isinstance(paras, dict):  # e.g. {"param": 1.0, ...}
            for key, val in paras.items():
                if replace or (not hasattr(self, key)):
                    setattr(self, key, val)

        elif isinstance(paras, list):  # e.g. ["--param", "1.0", ...]
            for idx in range(0,len(paras),2):
                if paras[idx][:2] == '--':
                    if replace or (not hasattr(self, key)):
                        setattr(self, paras[idx][2:], paras[idx+1])

        else: raise TypeError('Type of <paras> has to be <dict> or <list>')

    def import_paras(self, cfg_path):
        '''Cannot overwrite existing attributes.'''
        cfg = SnL.generic_load(cfg_path)
        cfg = self._flatten_(cfg)
        self.set_paras(cfg, replace=False)
    
    def import_training_paras(self, cfg_path):
        cfg = SnL.generic_load(cfg_path)

        exclusion = ['Description', 'device', 'n_epochs', 'early_stop', 'imsave_freq', 'result_image_amount', 'loadID', 'loadOPT', 'trainID']
        try:
            if self.loadOPT>0: exclusion.extend(['learning_rate', 'optim_type', 'scheduler'])
            setattr(self, 'Description', f'{self.Description}  |  >>(Continued training from result-{self.loadID})')
        except: pass
        for key in exclusion:
            try: cfg.pop(key)
            except :pass
        self.set_paras(cfg)

    def save(self, save_path):
        config = vars(self)
        SnL.generic_save(config, save_path)

    def _extractNone_(self, indict:dict):
        items = [(key, val) for key, val in indict.items() if val!=None]
        print(items)
        return dict(items)
    
    def _flatten_(self, indict:dict):
        items = []
        for key, val in indict.items():
            if isinstance(val, dict):
                items.extend(self._flatten_(val).items())
            else:
                items.append((key, val))
        outdict = dict(items)
        if len(items)!=len(outdict): warnings.warn("Repetition found in input dictionary has been reduced in output flatten dictionary, which may cause error somewhere.")
        return outdict
        