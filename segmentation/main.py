from segment import SlicSegment

if __name__ == '__main__':
  # put images of one class in source_dir/concept
  concept='bike'
  source_dir='../data/'

  slic = SlicSegment(source_dir, concept) 

  # Available param in create_patches
  # {'param1':[a,b,...], 'param2':[z,y,x,...], ...}
  # For instance {'n_segments':[15,50,80], 'compactness':[10,10,10]}
  image_numbers, dataset, patches = slic.create_patches()
  
  # returns: segmentations extract from the class, three level (8, 20, 30)
  # dataset: shape(num_seg, H, W, 3)  # mask on origal image
  # image_numbers: shape(num_seg), indicates which image the segmentation comes from
  # patches: shape(num_seg, H, W, 3)  # resize the masked image to model input size, default (229, 229), change the value in def _extract_patch(... image_shape=(229, 229))

  print("Segementation number: ", image_numbers.shape[0])
  # P.S. 10 image generate about 300 segmentions

  # image_show(dataset[20])
  # image_show(patches[20])