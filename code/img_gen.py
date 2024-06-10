import cv2, os, random
import numpy as np
def create_sample_64(n):
    """
    Create a sample of n images of size 64 x 64 with one of Waldos faces
    """
    
    Waldos = []
    
    for i in [3,5,6,9,14,16]:
        Waldo = cv2.imread(f'../data/src2/OnlyWaldoHeads/{i}.png')
        if Waldo.shape[2] == 3:
            Waldo = cv2.cvtColor(Waldo, cv2.COLOR_BGR2BGRA)
        
        Waldos.append(Waldo)
            
    imgs = []
    
    for i in range(n):
        path = random.choice(os.listdir('../data/src1/64/notwaldo'))
        img = cv2.imread(f'../data/src1/64/notwaldo/{path}')
        
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        Waldo = random.choice(Waldos)
        
        x = random.randint(0, 64 - Waldo.shape[1])
        y = random.randint(0, 64 - Waldo.shape[0])
        
                
        black_mask = np.all(Waldo[:, :, :3] == [0,0,0], axis=-1)
        
        Waldo[black_mask, 3] = 0
        
        alpha = Waldo[:,:,3] / 255.0
        inv_alpha = 1.0 - alpha
        
        for c in range(0, 3):
            img[y:y+Waldo.shape[0], x:x+Waldo.shape[1], c] = (alpha * Waldo[:,:,c] + inv_alpha * img[y:y+Waldo.shape[0], x:x+Waldo.shape[1], c])
        
        imgs.append(img)
        
    # save images
    
    for i, img in enumerate(imgs):
        cv2.imwrite(f'../data/selfmade/waldo/64_{i}.png', img)
        
    return imgs
    

def create_sample_128(n):
    """
    Create a sample of n images of size 128 x 128
    """
    Waldos = []
    
    for i in [3,5,6,9,14,16]:
        Waldo = cv2.imread(f'../data/src2/OnlyWaldoHeads/{i}.png')
        if Waldo.shape[2] == 3:
            Waldo = cv2.cvtColor(Waldo, cv2.COLOR_BGR2BGRA)
        
        Waldos.append(Waldo)
            
    imgs = []
    
    for i in range(n):
        path = random.choice(os.listdir('../data/src1/128/notwaldo'))
        img = cv2.imread(f'../data/src1/128/notwaldo/{path}')
        
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        Waldo = random.choice(Waldos)
        
        x = random.randint(0, 128 - Waldo.shape[1])
        y = random.randint(0, 128 - Waldo.shape[0])
        
                
        black_mask = np.all(Waldo[:, :, :3] == [0,0,0], axis=-1)
        
        Waldo[black_mask, 3] = 0
        
        alpha = Waldo[:,:,3] / 255.0
        inv_alpha = 1.0 - alpha
        
        for c in range(0, 3):
            img[y:y+Waldo.shape[0], x:x+Waldo.shape[1], c] = (alpha * Waldo[:,:,c] + inv_alpha * img[y:y+Waldo.shape[0], x:x+Waldo.shape[1], c])
        
        imgs.append(img)
        
    # save images
    
    for i, img in enumerate(imgs):
        cv2.imwrite(f'../data/selfmade/waldo/128_{i}.png', img)
        
    return imgs

def create_sample_256(n):
    """
    Create a sample of n images of size 256 x 256
    """
    Waldos = []
    
    for i in [3,5,6,9,14,16]:
        Waldo = cv2.imread(f'../data/src2/OnlyWaldoHeads/{i}.png')
        if Waldo.shape[2] == 3:
            Waldo = cv2.cvtColor(Waldo, cv2.COLOR_BGR2BGRA)
        
        Waldos.append(Waldo)
            
    imgs = []
    
    for i in range(n):
        path = random.choice(os.listdir('../data/src1/256/notwaldo'))
        img = cv2.imread(f'../data/src1/256/notwaldo/{path}')
        
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # Waldo = random.choice(Waldos)
        Waldo = Waldos[0]
        
        x = random.randint(0, 256 - Waldo.shape[1])
        y = random.randint(0, 256 - Waldo.shape[0])
        
                
        black_mask = np.all(Waldo[:, :, :3] == [0,0,0], axis=-1)
        
        Waldo[black_mask, 3] = 0
        
        alpha = Waldo[:,:,3] / 255.0
        inv_alpha = 1.0 - alpha
        
        for c in range(0, 3):
            img[y:y+Waldo.shape[0], x:x+Waldo.shape[1], c] = (alpha * Waldo[:,:,c] + inv_alpha * img[y:y+Waldo.shape[0], x:x+Waldo.shape[1], c])
        
        imgs.append(img)
        
    # save images
    
    for i, img in enumerate(imgs):
        cv2.imwrite(f'../data/selfmade/waldo/256_{i}.png', img)
        
    return imgs

def create_sample(size, n, remove_old=True):
    """
    Create a sample of n images of size size x size
    """
    if remove_old:
        for file in os.listdir('../data/selfmade/waldo'):
            os.remove(f'../data/selfmade/waldo/{file}')
    if size == 64:
        create_sample_64(n)
    elif size == 128:
        create_sample_128(n)
    elif size == 256:
        create_sample_256(n)
    else:
        print('Size not supported')
        

if __name__ == '__main__':
    create_sample(128, 10)