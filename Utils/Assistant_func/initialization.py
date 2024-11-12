import os
try: import Utils.Assistant_func.SnL as SnL
except: import Utils.Assistant_func.SnL as SnL

def Display_info(config, path, create_new=True):
    info = ""
    info += f"*\nInformation of configurations:"
    for ele in config:
        info += f"\n  {ele} : {config[ele]}"
    info += f"\n*\nInformation of pathways:"
    for ele in path:
        info += f"\n  {ele} : {path[ele]}"
    print("\n*"+info)
    
    # save information as .txt
    if create_new:
        SnL.save_(info, os.path.join(path['result_path'], 'info.txt'))

def Path_init(path_file, key, loadID=-1, create_new=True):
    path = SnL.generic_load(path_file)
    assert isinstance(path, dict)
    if loadID>0:
        path['load_config_path'], path['load_model_path'] = set_path(path, loadID, create_new=False)
    path['result_path'], path['model_path'] = set_path(path, key, create_new)
    return path

def set_path(path, key=0, create_new=True):
    IDX_PATH, IDX_MODEL = -1, -5  # model_path ends with .pth
    
    # initialize saving result path
    rsltpath = _string_modify(path['result_path'], IDX_PATH, str(key))
    if create_new:
        while os.path.isdir(rsltpath):
            key += 1
            rsltpath = _string_modify(path['result_path'], IDX_PATH, str(key))
        os.mkdir(rsltpath)

    # initialize model path
    mdlpath = _string_modify(path['model_path'], IDX_MODEL, str(key))
    if create_new:
        model = _split_by_folder(path['model_path'])[-1]
        modeldir = path['model_path'][:-len(model)]
        os.makedirs(modeldir, exist_ok=True)
    return rsltpath, mdlpath
    
def _string_modify(instring, index, words):
    string_list = list(instring)
    string_list[index] = words
    return ''.join(string_list)

def _split_by_folder(path: str):
    splitted = path.split('\\')
    if len(splitted)==1:
        splitted = path.split('/')
        if len(splitted)<=1:
            raise RuntimeError(f"Invalid path of {path}")
    return splitted
