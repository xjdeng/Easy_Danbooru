#from Easy_Image import imagesearch as ims
try:
    import deep_danbooru_model as ddm
except ImportError:
    from . import deep_danbooru_model as ddm
import numpy as np
import pandas as pd
import gc
from path import Path as path
from tqdm import tqdm

model = ddm.DeepDanbooruModel()
model.initialize()

with open("{}/besttags.txt".format(path(__file__).abspath().dirname()),'r') as f:
    txt = f.read()
tags = set(txt.split("\n"))

default_columns = ['file','mtime'] + list(tags)
dtypes = {d: np.uint8 for d in tags}
dtypes['mtime'] = np.float64

modeltags = set(model.tags)
for t in tags:
    if t not in modeltags:
        print(t)

def smartwalkfiles(start):
    files = path(start).files()
    for d in path(start).dirs():
        try:
            files += smartwalkfiles(d)
        except (TypeError, PermissionError):
            pass
    return files

def same_mtime(t1, t2):
    if not t1:
        return False
    if t1 == t2:
        return True
    diff = abs(t1 - t2)
    if diff % 3600 == 0:
        return True
    if diff < 0.001:
        return True
    return False

def smart_lookup(mtimes, name, mtime):
    tmp = mtimes.get((name, mtime), None)
    if tmp:
        return tmp
    tmp = mtimes.get((name, mtime + 3600), None)
    if tmp:
        return tmp
    return mtimes.get((name, mtime - 3600), None)

def vector(f, fpath, mtime):
    result = model.get_raw(f)
    if result is None:
        addition = [-1]*len(tags)
    else:
        #addition = [int(round(100*a)) for a in result]
        result = np.array([a for a,t in zip(result,model.tags) if t in tags])
        addition = np.round(100*result).astype(np.uint8)
    #return [[fpath, mtime] + addition]
    return [np.hstack((np.array([fpath, mtime]), addition))]

def run(start = "./", outfile = "danbooru.csv", batch = 40000):
    run_meta(vector, default_columns, outfile, start, batch)
    
def run_meta(func, columns, default_file, start = "./", batch = 1000, detect_moved = False):
    idxfile = "{}/{}".format(start, default_file)
    #addition = pd.DataFrame(columns=columns)
    add_idx, add = [],[]
    existing = None
    def save(ex, ad):
        addd_idx, addd = ad
        ad = pd.DataFrame(addd, columns=columns, index=addd_idx)
        ad = ad.astype(dtypes)
        print("Saving results")
        gc.collect()
        #ex = ex.append(ad, sort=True)
        ex = pd.concat([ex, ad], ignore_index=True, sort=True)
        gc.collect()
        oldpaths = list(ex['file'])
        ex['file'] = [p.replace(start, "./") for p in ex['file']]
        ex[columns].to_csv(idxfile)
        ex['file'] = oldpaths
    try:
        filequeue = set([str(f).replace("\\","/") for f in smartwalkfiles(start)])
        try:
            existing = pd.read_csv(idxfile, index_col = 0, low_memory=True, dtype=dtypes)
            existing.index = list(range(0, len(existing)))
            existing.columns = columns
            existing['file'] = [p.replace("./",start) for p in existing['file']]
            if len(existing.index) == 0:
                idx = 0
            else:
                idx = 1 + max(existing.index)
            gc.collect()
            lookup = {f.replace("./",start):s for (f,s) in zip(existing['file'], existing['mtime'])}
            remove_keys = []
            remove_filequeue = []
            mtimes = None
            for f in lookup.keys():
                if f not in filequeue:
                    if not mtimes:
                        
                        mtimes = {}
                        for f2 in filequeue:
                            fp = path(f2)
                            mtimes[(fp.name, fp.mtime)] = f2
                    pathf = path(f)
                    #f2 = mtimes.get((pathf.name, lookup[f]), None)
                    f2 = smart_lookup(mtimes, pathf.name, lookup[f])
                    if f2 is not None:
                        print("Moved file detected: {}".format(f))
                        if detect_moved:
                            tmp = existing[existing['file']==f]
                            for index in tmp.index:
                                newrow = list(tmp.loc[index])
                                newrow[0] = f2
                                #addition.loc[idx] = newrow
                                add_idx.append(idx)
                                add.append(newrow)
                                idx += 1
                        remove_filequeue.append(f2)
                    remove_keys.append(f)
                    print("{} no longer found, deleting from database".format(f))
            remove_keys = set(remove_keys)
            remove_idx = []
            print("Stage1")
            for idx2 in existing.index:
                f = existing['file'].loc[idx2]
                try:
                    if f in remove_keys:
                        print("Looking at file: " + f)
                        remove_idx.append(idx2)
                        print("Deleting")
                        try:
                            print("Looking up")
                            del lookup[f]
                        except KeyError:
                            print("KeyError")
                except TypeError:
                    remove_idx.append(idx)
                    print("TypeError")
            print("Stage2")
            existing.drop(existing.index[remove_idx], inplace=True)
            print("Stage3")
            for f2 in remove_filequeue:
                try:
                    filequeue.remove(f2)
                except KeyError:
                    print("Error: failed to remove {} from filequeue".format(f2))
        except IOError:
            lookup = {}
            existing = pd.DataFrame(columns=columns)
            existing = existing.astype(dtypes)
            idx = 0
        j = 0
        print(len(filequeue))
        existing_files = set(existing['file'])
        for i,f0 in enumerate(tqdm(filequeue)):
            f = path(f0)
            mtime = f.mtime
            fpath = str(f).replace("\\","/")
            #print(fpath)
            if not same_mtime(lookup.get(fpath), mtime):#lookup.get(fpath) != mtime:
                #print("Adding: {}".format(fpath))
                #print(lookup.get(fpath), mtime)
                j += 1
                #print(fpath)
                #print(existing.tail())
                if fpath in existing_files:
                    existing.drop(existing.loc[existing['file'] == fpath].index, inplace=True)
                #print(existing.tail())
                #Begin snippet
                try:
                    additions = func(f, fpath, mtime)
                    for ad in additions:
                        #addition.loc[idx] = add
                        add_idx.append(idx)
                        add.append(ad)
                        idx += 1
                except PermissionError:                    
                    pass#print("Permission Error, skipping")
                #End snippet
            else:
    
                pass#print("Skipping existing file")
            #print("{} out of {} files completed".format(1+i, len(filequeue)))
            if (j+1) % batch == 0:
                #print("Appending current batch")
                gc.collect()
                addition = pd.DataFrame(add, columns=columns, index=add_idx)
                addition = addition.astype(dtypes)
                #existing = existing.append(addition, sort=True)
                existing = pd.concat([existing, addition], ignore_index=True, sort = True)
                existing_files = set(existing['file'])
                gc.collect()
                #print("Saving results")
                existing[columns].to_csv(idxfile)
                gc.collect()
                #addition = pd.DataFrame(columns=columns, index=add_idx)
                add_idx, add = [], []
                gc.collect()
                j += 1            
        save(existing, (add_idx, add))
    except KeyboardInterrupt as e:
        print(e)
        save(existing, (add_idx, add))
    #except Exception as e:
    #    print(e)
    #    print("Outer Exception")
    #    save(existing, (add_idx, add))
    
def run_hdf(func, columns, default_file, start = "./", batch = 1000):
    idxfile = "{}/{}".format(start, default_file)
    #addition = pd.DataFrame(columns=columns)
    add_idx, add = [],[]
    existing = None
    def save(ex, ad):
        addd_idx, addd = ad
        ad = pd.DataFrame(addd, columns=columns, index=addd_idx)
        ad = ad.astype(dtypes)
        print("Saving results")
        gc.collect()
        #ex = ex.append(ad, sort=True)
        ex = pd.concat([ex, ad], ignore_index=True, sort=True)
        gc.collect()
        oldpaths = list(ex['file'])
        ex['file'] = [p.replace(start, "./") for p in ex['file']]
        ex[columns].to_hdf(idxfile, key="default")
        ex['file'] = oldpaths
    try:
        filequeue = set([str(f).replace("\\","/") for f in smartwalkfiles(start)])
        try:
            existing = pd.read_hdf(idxfile, index_col = 0, low_memory=True, dtype=dtypes)
            existing.index = list(range(0, len(existing)))
            existing.columns = columns
            existing['file'] = [p.replace("./",start) for p in existing['file']]
            if len(existing.index) == 0:
                idx = 0
            else:
                idx = 1 + max(existing.index)
            gc.collect()
            lookup = {f.replace("./",start):s for (f,s) in zip(existing['file'], existing['mtime'])}
            remove_keys = []
            remove_filequeue = []
            mtimes = None
            for f in lookup.keys():
                if f not in filequeue:
                    if not mtimes:
                        
                        mtimes = {}
                        for f2 in filequeue:
                            fp = path(f2)
                            mtimes[(fp.name, fp.mtime)] = f2
                    pathf = path(f)
                    #f2 = mtimes.get((pathf.name, lookup[f]), None)
                    f2 = smart_lookup(mtimes, pathf.name, lookup[f])
                    if f2 is not None:
                        print("Moved file detected: {}".format(f))
                        tmp = existing[existing['file']==f]
                        for index in tmp.index:
                            newrow = list(tmp.loc[index])
                            newrow[0] = f2
                            #addition.loc[idx] = newrow
                            add_idx.append(idx)
                            add.append(newrow)
                            idx += 1
                        remove_filequeue.append(f2)
                    remove_keys.append(f)
                    print("{} no longer found, deleting from database".format(f))
            remove_keys = set(remove_keys)
            remove_idx = []
            print("Stage1")
            for idx2 in existing.index:
                f = existing['file'].loc[idx2]
                try:
                    if f in remove_keys:
                        print("Looking at file: " + f)
                        remove_idx.append(idx2)
                        print("Deleting")
                        try:
                            print("Looking up")
                            del lookup[f]
                        except KeyError:
                            print("KeyError")
                except TypeError:
                    remove_idx.append(idx)
                    print("TypeError")
            print("Stage2")
            existing.drop(existing.index[remove_idx], inplace=True)
            print("Stage3")
            for f2 in remove_filequeue:
                filequeue.remove(f2)
        except IOError:
            lookup = {}
            existing = pd.DataFrame(columns=columns)
            existing = existing.astype(dtypes)
            idx = 0
        j = 0
        print(len(filequeue))
        existing_files = set(existing['file'])
        for i,f0 in enumerate(filequeue):
            f = path(f0)
            mtime = f.mtime
            fpath = str(f).replace("\\","/")
            print(fpath)
            if not same_mtime(lookup.get(fpath), mtime):#lookup.get(fpath) != mtime:
                #print("Adding: {}".format(fpath))
                #print(lookup.get(fpath), mtime)
                j += 1
                #print(fpath)
                #print(existing.tail())
                if fpath in existing_files:
                    existing.drop(existing.loc[existing['file'] == fpath].index, inplace=True)
                #print(existing.tail())
                #Begin snippet
                try:
                    additions = func(f, fpath, mtime)
                    for ad in additions:
                        #addition.loc[idx] = add
                        add_idx.append(idx)
                        add.append(ad)
                        idx += 1
                except PermissionError:                    
                    print("Permission Error, skipping")
                #End snippet
            else:
    
                print("Skipping existing file")
            print("{} out of {} files completed".format(1+i, len(filequeue)))
            if (j+1) % batch == 0:
                print("Appending current batch")
                gc.collect()
                addition = pd.DataFrame(add, columns=columns, index=add_idx)
                addition = addition.astype(dtypes)
                existing = existing.append(addition, sort=True)
                existing_files = set(existing['file'])
                gc.collect()
                print("Saving results")
                existing[columns].to_hdf(idxfile, key="default")
                gc.collect()
                #addition = pd.DataFrame(columns=columns, index=add_idx)
                add_idx, add = [], []
                gc.collect()
                j += 1            
        save(existing, (add_idx, add))
    except KeyboardInterrupt as e:
        print(e)
        save(existing, (add_idx, add))
    #except Exception as e:
    #    print(e)
    #    print("Outer Exception")
    #    save(existing, (add_idx, add))