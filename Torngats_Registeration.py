#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 10:40:03 2022

@author: sami
"""


import os, sys
import numpy as np
from pystackreg import StackReg
import pystackreg 
from skimage import io
from deepdish.io import load,save
from skimage import transform, io, exposure
from matplotlib import pyplot as plt
# NCT = r'C:\TORNGATS\CODE'
# if NCT not in sys.path:
#     sys.path.insert(0, NCT)
from nct1 import contrast_correction, multiplot, image_show
from natsort import natsorted
#%%

def overlay_images(imgs, equalize=False, aggregator=np.mean):
    '''
    

    Parameters
    ----------
    plot the overlay images
    
    imgs : list of image or 3d array
        DESCRIPTION.
        list of image to plot as a overlay image
        
    equalize : Boolean, optional
        DESCRIPTION. The default is False.
        To enhance the visibility (contrast) of the image 
    aggregator : TYPE, optional
        DESCRIPTION. The default is np.mean.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = np.stack(imgs, axis=0)

    return aggregator(imgs, axis=0)

#%%
def composite_images(imgs, equalize=False, aggregator=np.mean):
    '''
    
    
    Parameters
    ----------
    imgs : TYPE
        DESCRIPTION.
        list of two images to show as a single image in RGB axis
    equalize : TYPE, optional
        DESCRIPTION. The default is False.
        enhance the contrast
    aggregator : TYPE, optional
        DESCRIPTION. The default is np.mean.

    Returns
    -------
    imgs : TYPE
        DESCRIPTION.

    '''

    if equalize:
        imgs = [exposure.equalize_hist(img) for img in imgs]

    imgs = [img / img.max() for img in imgs]

    if len(imgs) < 3:
        imgs += [np.zeros(shape=imgs[0].shape)] * (3-len(imgs))

    imgs = np.dstack(imgs)

    return imgs
#%%
def show_transformation(tmat, ax=None):
    '''
    plot the transformation matrix 

    Parameters
    ----------
    tmat : TYPE
        DESCRIPTION.
        transformation matrix
        
    ax : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    if ax is None:
        _, ax = plt.subplots()
    p = np.array([[1,120,1], [1,1,1], [250, 1, 1], [250,120,1], [1,120,1]])
    ax.plot(p[:, 0], p[:,1])
    q=np.dot(p, tmat.T)
    ax.plot(q[:, 0], q[:,1])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend(['Original image', 'transformed image'])

#%%
def registeration(unreg, registeration_type = 'single',transformation_type = 'all',
                  reference = 'previous', show_overlay= False , filename=None , opt='list'):
    '''
    register sequence and single image
    
    

    Parameters
    ----------
    unreg : ndarray 
        -for single registeration the input data should be as a list i.e. [ref, mov]
        -for stack registeration the input data should be 3d array which the first index is the frame number
            
    registeration_type : string , optional
        DESCRIPTION. The default is 'single'.
        single or stack for single-registeration and stack-registeration, respectively.
        
    transformation_type : string, optional
        DESCRIPTION. The default is 'all'.
        the value can be :
            'all'/'TRANSLATION' /'RIGID_BODY'/'SCALED_ROTATION'/'AFFINE'/'BILINEAR'
        
    reference : string, optional
        DESCRIPTION. The default is 'previous'.
        the value can be:
            'previous'/ 'first'/'mean'
    show_overlay : BOOL, optional
        DESCRIPTION. The default is False.
        plot the overlay images
    
    filename : string, optional
        The default is 'None'.
        to save the output in a file set the filename.
        the file saves in current work directory.
    
    opt : string
        the default value is 'list'.
        it can be 'list' or 'dict'.

    Returns
    -------
    regs : Dict./list of ndarray
        registered images
        
    tmats : dict./list of ndarray
        transformation matrices
    '''
    
    #--------------------------------------------------------------------------
    
    if transformation_type == 'all':
        transformations = {
        'TRANSLATION': StackReg.TRANSLATION,
        'RIGID_BODY': StackReg.RIGID_BODY,
        'SCALED_ROTATION': StackReg.SCALED_ROTATION,
        'AFFINE': StackReg.AFFINE,
        'BILINEAR': StackReg.BILINEAR
        }
        
        
    elif transformation_type =='TRANSLATION':
        transformations = {'TRANSLATION': StackReg.TRANSLATION}
        
    elif transformation_type =='RIGID_BODY':
        transformations = {'RIGID_BODY': StackReg.RIGID_BODY}
        
    elif transformation_type =='SCALED_ROTATION':
        transformations = {'SCALED_ROTATION': StackReg.SCALED_ROTATION}
   
    elif transformation_type =='AFFINE':
        transformations = {'AFFINE': StackReg.AFFINE}
   
    elif transformation_type =='BILINEAR':
        transformations = {'BILINEAR': StackReg.BILINEAR}
        
    #--------------------------------STACK REGISTERATION-----------------------
    
    if opt == 'dict':
        regs , tmats = {},{}
    elif opt =='list':
        regs , tmats = [], []
    
    if registeration_type == 'stack':
        
        for i, (name, tf) in enumerate(transformations.items()):
            sr = StackReg(tf)
        
            reference = 'first' if name == 'BILINEAR' else reference
            tmat = sr.register_stack(unreg, axis=0, reference=reference, verbose=True)
            reg = sr.transform_stack(unreg)
            
            # regs.append(reg)
            # tmats.append(tmat)
            if opt == 'dict':
                regs[name] = reg
                tmats[name] = tmat
            elif opt =='list':
                regs.append(reg)
                tmats.append(tmat)
            
    #---------------------------SINGLE REGISTERATION---------------------------
    elif registeration_type =='single':
        ref, mov= unreg[0], unreg[1]
        for i, (name, tf) in enumerate(transformations.items()):
            
            sr = StackReg(tf)
            reg = sr.register_transform(ref, mov)
            tmat = sr.get_matrix()
            # tpts = sr.get_points()
            
            if opt == 'dict':
                regs[name] = reg
                tmats[name] = tmat
            elif opt =='list':
                regs.append(reg)
                tmats.append(tmat)
            
    #----------------------------PLOT the overlay images-----------------------            
    if show_overlay:
        f, ax = plt.subplots(2, int(np.ceil((len(transformations)+1)/2)), figsize=(20, 12))
        ax = ax.ravel()
        
        ax[0].imshow(overlay_images(unreg, aggregator=np.mean), cmap='gray')
        ax[0].set_title('Original (overlay)')
        ax[0].axis('off')
        
        for i, (name, tf) in enumerate(transformations.items()):
            ax[i+1].imshow(overlay_images(reg, aggregator=np.mean), cmap='gray')
            ax[i+1].set_title(name + ' (overlay)')
            ax[i+1].axis('off')
    #------------------------------save OUTPUTS--------------------------------
    
    if filename is not None:
        pth = os.path.join(os.getcwd(),(filename+'_'+transformation_type+'_'+reference+'.h5'))
        d ={'regs':regs, 'tmats':tmats}
        save(pth,d, compression=('blosc',9))
            
    return regs, tmats
    #--------------------------------------------------------------------------

#%%
def plot_registered (data , reg_idxs, x,y,allInOne =False,data2=None, regUnreg =False ):
    
    if allInOne:
        imgs=[]
        for i in reg_idxs:
            print(i)
            imgs.append(overlay_images([data[i],data[i+1]]))
        
        
        multiplot(imgs,x,y)
    elif regUnreg:
        for i in reg_idxs:
            multiplot([overlay_images([data[i],data[i+1]]), 
            overlay_images([data2[i],data2[i+1]])])
            
#%%
def load_transformation(unreg, tmats, transformation_type, show_overlay=False):
    '''
    apply transformations on unregistered sequence
    it works for single or sequence registeration
    the input in both case are dict. 
    in case of single kind of transformation (Translation, or Rigidbody or scaled-rotation or affine)
    the input matrix can convert to the desire shape.

    Parameters
    ----------
    unreg : 3d array
        dict. of stacked images
    tmats : dict. of transformation matrices
    
    transformation_type : STRING
        the value can be :
            'all'/'TRANSLATION' /'RIGID_BODY'/'SCALED_ROTATION'/'AFFINE'/'BILINEAR'.
            
    show_overlay : BOOL
        plot the overlay matrix
        
    Returns
    -------
    regs : ndarray
        registered images

    '''
#%%
#------------------if the tmats is not a dict ---------------------------------    
    # if not (isinstance( tmats, dict) ) :
    #     #---------------------convert to desire style--------------------------
    #     #if the input matrix in not a 3d array
    #     if len(tmats.shape )== 2:
    #         rows, cols = tmats.shape
    #         tmats = tmats.reshape(1,rows, cols)
            
    #     tmp = {}
    #     # convert to dictionary
    #     tmp[transformation_type] = tmats
    #     tmats = tmp
    #     # print('tmats type:', type(tmats))
     
            
    if len(unreg.shape) > len(tmats.shape) :
        
        frames, rows, cols = unreg.shape
        if len(tmats.shape)==2:
            r, c = tmats.shape
            
            tmats = np.repeat(tmats.reshape(1,r,c) ,axis=0, repeats=frames)
        
    if len(unreg.shape)==2 and len(tmats.shape)==2 :
        rows, cols = unreg.shape
        unreg = unreg.reshape(1,rows, cols)
        r , c = tmats.shape
        tmats = tmats.reshape(1, r , c) 
    
#--------------------------- make transformation dict. -------------------------         
    if transformation_type == 'all':
        transformations = {
        'TRANSLATION': StackReg.TRANSLATION,
        'RIGID_BODY': StackReg.RIGID_BODY,
        'SCALED_ROTATION': StackReg.SCALED_ROTATION,
        'AFFINE': StackReg.AFFINE,
        'BILINEAR': StackReg.BILINEAR
        }
        
        
    elif transformation_type =='TRANSLATION':
        transformations = {'TRANSLATION': StackReg.TRANSLATION}
        
    elif transformation_type =='RIGID_BODY':
        transformations = {'RIGID_BODY': StackReg.RIGID_BODY}
        
    elif transformation_type =='SCALED_ROTATION':
        transformations = {'SCALED_ROTATION': StackReg.SCALED_ROTATION}
   
    elif transformation_type =='AFFINE':
        transformations = {'AFFINE': StackReg.AFFINE}
   
    elif transformation_type =='BILINEAR':
        transformations = {'BILINEAR': StackReg.BILINEAR}
    
    print(transformations)
#-----------------------------apply transformations------------------------------------
    regs =[]
    for i, (name, tf) in enumerate(transformations.items()):
    
        if name == 'BILINEAR':
            # Bilinear transformation is not an affine transformation, we can't use the transformation matrix here
            continue
    
        # copy the unregistered image
        reg = unreg.copy()
    
        for i_img in range(unreg.shape[0]):
            # get skimage's AffineTransform object
            tform = transform.AffineTransform(matrix=tmats[i_img, :, :])
    
            # transform image using the saved transformation matrix
            reg[i_img, :, :] = transform.warp(reg[i_img, :, :], tform)
            
        regs.append(reg)
#------------------------------plot overlay images----------------------------------    
    if show_overlay:
        f, ax = plt.subplots(2, int(np.ceil((len(transformations)+1)/2)), figsize=(20, 12))
        ax = ax.ravel()
        
        ax[0].imshow(overlay_images(unreg, aggregator=np.mean), cmap='gray')
        ax[0].set_title('Original (overlay)')
        ax[0].axis('off')
        
        for i, (name, tf) in enumerate(transformations.items()):
            ax[i+1].imshow(overlay_images(reg, aggregator=np.mean), cmap='gray')
            ax[i+1].set_title(name + ' (overlay)')
            ax[i+1].axis('off')
    return regs

#%%
def compare_tmats(tmats, comp_with='previous', flatten = True, range_value=10):
    '''
    

    Parameters
    ----------
    tmats : ndarray- 3d
        DESCRIPTION. Transformation matrix for an stacked images
        
    comp_with : STRING , optional
        DESCRIPTION. The default is 'previous'.
        
    flatten : bool, optional
        DESCRIPTION. The default is True.
        
    range_value : TYPE, optional
        DESCRIPTION. The default is 10.
        the value which is >= range value will be returned.

    Returns
    -------
    subt_ftmats : ndarray
        the subtracted transformation correspond to comp_with value.
        
    range_value_list : list
        DESCRIPTION.
        List of image indexes which is higher than range_value
        
    '''
    
    frames, rows, cols = tmats.shape
    
    if flatten and comp_with == 'previous':
        tmats = np.round(tmats,2)
        tmats = np.asarray([t.flatten() for t in tmats])
        subt_ftmats=[]
        # subtract flatten transformation matrix from previous rows to find 
        #big movement according to range_value
        for i in range(frames):
            if i!= frames -1:  
                subt_ftmats.append(tmats[i]-tmats[i+1])
        subt_ftmats = np.asarray(subt_ftmats)
        
        # check all the columns for the desire indexes
        range_value_list = []
        for i in range(rows*cols):
            col=abs(subt_ftmats[:,i])
            mv = np.asarray( [ [n,i] for n,i in enumerate(col) if i>=range_value ]).astype(int)
            if len(mv)!=0:
                range_value_list.append(mv )
        # take union of the found index for final list 
        union_list=[]
        for i in range(len(range_value_list)):
            union_list = union_list+ list(range_value_list[i][:,0])
        union_list = natsorted(list(set(union_list)))
        # final list of indexes 
        range_value_list = union_list
    elif flatten and comp_with == 'first':
        print('COME BACK SOON TO DO! ')
        
    return subt_ftmats, range_value_list   
#%%
def resgisteration_selextedIndex(data,mov_mat, transformation_type = 'TRANSLATION', reference = 'previous'):
    '''
    registeration images based on defined indeces    
    this function apply the transformation on defined index and also the same 
    transformation is applied to the images with no movement.
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    mov_mat : list
        DESCRIPTION. list of index which is in defined range
        
    transformation_type : string, optional
        DESCRIPTION. The default is 'TRANSLATION'.
        it can be one of these value: 
            'TRANSLATION' /'RIGID_BODY'/'SCALED_ROTATION'/'AFFINE'/'BILINEAR'.
            
    reference : string, optional
        DESCRIPTION. The default is 'previous'.
        it can be one of these values:
            "previous"/'first'/'mean'

    Returns
    -------
    new_seq : ndarray
        DESCRIPTION.
        registered images seq.
    '''
    frames,rows, cols = data.shape
    for idx, i in enumerate(mov_mat):
        
        print( idx, i)
        # create new_seq 
        if idx ==0 or len(mov_mat)==1:
            new_seq= data[:i+1]
            
        # if first index AND if not reach to the list of index
        
        if idx ==0 and idx != len(mov_mat)-1:
            
            # single-registeration 
            reg, tmat = registeration([data[i], data[i+1]], registeration_type='single',
                                             transformation_type= transformation_type,
                                             show_overlay= False, opt = 'list')
            new_seq = np.vstack((new_seq, reg[0].reshape(1,rows,cols)))
            
            #compare the index value in list to take a disition for next step
            # check if there is any frame to register based on computed tmats 
            if  (mov_mat[idx+1])- (i+1) >0 :
                reg = load_transformation(data[i+2: mov_mat[idx+1]+1],tmat[0],
                                                  transformation_type=transformation_type, 
                                                  show_overlay = False)
                new_seq = np.vstack((new_seq, reg[0]))

        # if reach to the end of list 
        elif idx == len(mov_mat)-1:
            reg, tmat = registeration([new_seq[-1], data[i+1] ], registeration_type='single',
                                                     transformation_type= transformation_type,
                                                     show_overlay= False , opt = 'list')
            
            new_seq = np.vstack((new_seq, reg[0].reshape(1,rows,cols)))
            # if i <= len(data)-1:
            reg = load_transformation(data[i+2: ],tmat[0],
                                              transformation_type=transformation_type,
                                              show_overlay= False)
            new_seq = np.vstack((new_seq, reg[0]))
            
            
        # if not reached to the end and not the begining the list
        else:
            

            reg, tmat = registeration([new_seq[-1], data[i+1]] , registeration_type='single',
                                                     transformation_type= transformation_type
                                                     ,show_overlay=False, opt = 'list')
            new_seq = np.vstack((new_seq, reg[0].reshape(1,rows,cols)))


            if  (mov_mat[idx+1])- (i+1) >0 :
                # if len(tmat[0].shape) ==2 :
                #     tmat = tmat[0]
                #     r, c = tmat.shape
                #     tmat = tmat.reshape(1, r,c)
                reg = load_transformation(data[i+2: mov_mat[idx+1]+1],tmat[0],
                                                  transformation_type=transformation_type,
                                                  show_overlay=False)
                new_seq = np.vstack((new_seq, reg[0]))
    return new_seq

#%%
def registeration_based_movement(data, tmats= None,transformation_type = 'TRANSLATION' , 
                                 reference = 'previous',range_value =1 ):
    '''
    register data sequence based on available/ not availble transformation matrix
    if transformation matrix is not available it will be computed 
    

    Parameters
    ----------
    data : ndarray
        DESCRIPTION.
    tmats : ndarray, optional
        DESCRIPTION. The default is None.
        
    transformation_type : TYPE, optional
        DESCRIPTION. The default is 'TRANSLATION'.
        
    reference : TYPE, optional
        DESCRIPTION. The default is 'previous'.
        
    range_value : int , optional
        DESCRIPTION. The default is 1.
        based on range_value the indexes are chosen
    Returns
    -------
    newseq : ndarray
        DESCRIPTION.
        registered seqence

    '''
    
    if tmats is None:
        regs, tmats = registeration(data,registeration_type='stack',transformation_type=transformation_type,
                      reference= reference,show_overlay=False, opt='list')
        tmats= tmats[0]
    _, mov_mat = compare_tmats(tmats, comp_with='previous',flatten= True,range_value= range_value)
    
    
    newseq = resgisteration_selextedIndex(data,mov_mat,  transformation_type= transformation_type,
                                 reference=reference)
    
    return newseq
#%%
if __name__ =='__main__':
    print()    