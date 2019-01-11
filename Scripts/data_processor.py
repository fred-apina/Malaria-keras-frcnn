from imports import *
from utils import seed, resize_and_save

def dataSplitter(input_data_dir,output_data_dir,file_format,SIZE=255):
    classes = os.listdir(input_data_dir)

    for single_class in classes:
        class_path = os.path.join(input_data_dir,single_class)
        filenames = os.listdir(class_path)
        filenames = [os.path.join(class_path, f) for f in filenames if f.endswith(file_format)]

        # Split the images in 'data' into 85% train, 10% dev and 5% test
        # Make sure to always shuffle with a fixed seed so that the split is reproducible
        seed()
        filenames.sort()
        random.shuffle(filenames)

        split_1 = int(0.85 * len(filenames))
        split_2 = int(0.95 * len(filenames))
        train_filenames = filenames[:split_1]
        val_filenames = filenames[split_1:split_2]
        test_filenames = filenames[split_2:]

        filenames = {'train': train_filenames,
                     'val': val_filenames,
                     'test': test_filenames}

        if not os.path.exists(output_data_dir):
            print("{} folder does not exist".format(output_data_dir))
        else:
            for split in ['train', 'val', 'test']:
                output_dir_split = os.path.join(output_data_dir + "/"+split,single_class)
                if not os.path.exists(output_dir_split):
                    print("Warning: dir {} does not exists".format(output_dir_split))

                print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
                for filename in filenames[split]:
                    resize_and_save(filename, output_dir_split, size=SIZE)


def get_transform():
    #transform must be diferent in diffent phase[train,val,test]
    transform = {
    'train': transforms.Compose([
        #resize image on 255x255
        transforms.Resize((255,255)),
        #random crop with 244x244
        transforms.RandomCrop((224,224)),
        #horizantalflip and vertical flip
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #transfom image from numpy array to tensors
        transforms.ToTensor(),
        #normalize by mean 0.485 and std 0.22.. in each chanel of image (3 times since our image is RGB image)
        #resoan has been used in pretrained model but you can find mean and std if you have many images  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        #val center croping
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    }
    return transform


#method to load data from the output directory created
def dataLoaders(path, mode="train", batchsize=4):
    #before loading data we should transfom
    transform = get_transform()
   
   #check the mode /phase  
    if mode=="train":
        files=['train', 'val']
        shuffle=True
    else:
        files = [mode]
        shuffle = False
        
#add the data list using the dataset method from torchvision,transfom the datafiles on data 
    data = {x: torchvision.datasets.ImageFolder(os.path.join(path, x),transform[x]) for x in files}
    #load datalist into a dataloader from util method with batch sizes and shuffle
    dataloaders = {x: torch.utils.data.DataLoader(data[x], 
                                                batch_size=batchsize, 
                                                shuffle=True) 
                    for x in files}
    #put in variable the sizes and classname                
    data_sizes = {x: len(data[x]) for x in files}
    class_names = data[mode].classes

    return dataloaders, class_names, data_sizes


def get_images_list(dataloader):
    images = [x for x,_ in dataloader.dataset.imgs]
    return images

def load_results(exp_name, dtype="val", result_path="../Results/"):
    full_path=result_path+dtype+"/"+exp_name
    y_pred=np.load(full_path+"predictions.npy")
    y_true =np.load(full_path+"correct_labels.npy")
    prob =np.load(full_path+"pred_probability.npy")
    return y_pred, y_true, prob


if __name__ == '__main__':
    #dataSplitter("../input","../data",'.jpg',255)
    #loaders,class_names,_ = dataLoaders(path="../data")
    #get_images_list(loaders["val"])

    y_pred, y_true, prob=load_results("_malaria_resnet_")
    print(prob)

    