from PIL import Image
from os import listdir, scandir
import glob

train_inpath = "imageset/train"
test_inpath = "imageset/test"
train_images_outpath = "data/train_images"
test_images_outpath = "data/test_images"
train_labels_outpath = "data/train_labels"
test_labels_outpath = "data/test_labels"
face = "Face"
motor = "Motor"

def create_matrixes_and_labels(path, out_images, out_labels):
    list_folders = [f.path for f in scandir(path) if f.is_dir()]

    files = []
    for name in list_folders:
        files += glob.glob(name + "/*.png") + glob.glob(name + "/*.jpg")

    out_images_file = open(out_images, 'x')
    out_labels_file = open(out_labels, 'x')

    for f in range(len(files)):
        if (f > 0):
            out_images_file.write('\n')
            out_labels_file.write('\n')
        img = Image.open(files[f]).convert("L")

        width, height = img.size;

        # *** Change image size *** #
        
        asp_rat = width/height;

        new_width = 213;
        new_height = 160;

        new_rat = new_width/new_height;

        if (new_rat == asp_rat):
            img = img.resize((new_width, new_height), Image.ANTIALIAS); 
        else:
            new_width = round(new_height * asp_rat);
            img = img.resize((new_width, new_height), Image.ANTIALIAS);

        # img.save(out_path + f[len(in_path):])

        # *** end of Change image size *** #

        px = list(img.getdata())

        for i in range (len(px)):
            if (i > 0):
                out_images_file.write(';')
            out_images_file.write(str(px[i]))
        if (files[f].find(face) != -1):
            out_labels_file.write(face)
        else:
            out_labels_file.write(motor)
    out_images_file.close()
    out_labels_file.close()


create_matrixes_and_labels(train_inpath, train_images_outpath, train_labels_outpath)
create_matrixes_and_labels(test_inpath, test_images_outpath, test_labels_outpath)