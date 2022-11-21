// #include <algorithm>

#include "Dataset_handle.h"
#include "Network.h"
#include "Util.h"
#include <string.h>
#include <setjmp.h>

/*
g++ -lm -O2 -Wall -std=c++11 *.cpp -o main
g++ -lm -O2 -Wall -std=c++11 -Xpreprocessor -fopenmp openMPTest.cpp -o main -lomp
conda activate /Users/edvaldojunior/Documents/Python/tf_env
*/

const std::string path = "data/";
const std::string train_dataset_path = path + "train_images";
const std::string train_labels_path  = path + "train_labels";
const std::string test_dataset_path  = path + "test_images";
const std::string test_labels_path   = path + "test_labels";

const std::vector<std::string> options_str = {
    "Quit",
    "Create network",
    "Create network from file",
    "Show network's description",
    "Show convolutional kernel",
    "Save network",
    "Train network",
    "Test network",
    "Show layer potentials"
};

Network net;
vvvi train_dataset, test_dataset;
vi train_labels, test_labels;
int selected = 6;
int Network::img_input_width = 213;
int Network::img_input_height = 160;

void show_options();
bool switch_options();
bool quit();
bool create_net();
bool create_net_from_file();
bool show_description();
bool show_kernel();
bool save_net();
bool train_net();
bool test_net();
bool show_layer_potentials();
bool wrong_op();
void shuffle_arrays(vvvi& v_imgs, vi& v_lbls);

std::vector<bool (*)()> options_func = {
    quit,
    create_net,
    create_net_from_file,
    show_description,
    show_kernel,
    save_net,
    train_net,
    test_net,
    show_layer_potentials,
    wrong_op
};

int main() {

    bool training = true, testing = true;
    const std::string net_name = "out/net_L8";
    int train_n_imgs = 120000;//406;
    int test_n_imgs = 591;//591;
    std::string train_name("training_params.csv");
    std::string test_name("testing_params.csv");

    train_dataset = read_images(train_dataset_path, Network::img_input_height, 
                                Network::img_input_width);
    test_dataset = read_images(test_dataset_path, Network::img_input_height,
                               Network::img_input_width);
    // printf("\nRead images: done\n");

    train_labels = read_labels(train_labels_path);
    test_labels = read_labels(test_labels_path);

    // printf("\nRead labels: done\n");

    shuffle_arrays(train_dataset, train_labels);
    shuffle_arrays(test_dataset, test_labels);

    // printf("\nShuffle arrays: done\n");

    // bool loop = true;
    // while (loop) { show_options(); loop = switch_options(); }

    time_t start;
    time(&start);

    if (training) {
        net.save("net_B4");
        net.unsupervised_learning(train_dataset, train_labels, train_name, train_n_imgs);
        net.save("net_L8");
    } else { net = Network(net_name); }
    if (testing)
        net.unsupervised_testing(train_dataset, train_labels, test_dataset, test_labels,
                                 train_name, test_name, train_n_imgs, test_n_imgs);
    print_time_and_text(start, "Total time:");
    
    return 0;
}

void shuffle_arrays(vvvi& dataset, vi& v_lbls) {

    vi indexes(dataset.size()), cp_v_lbls(v_lbls);
    vvvi cp_v_imgs(dataset);

    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

    for (unsigned i = 0; i < indexes.size(); ++i) indexes[i] = (i);
    std::shuffle(indexes.begin(), indexes.end(), rng);
    
    for (unsigned i = 0; i < indexes.size(); ++i) {
        dataset[i] = cp_v_imgs[indexes[i]];
        v_lbls[i] = cp_v_lbls[indexes[i]];
    }
}

void wait() { printf("\nPress Enter.\n\n"); getchar(); getchar(); }

void show_options() {

    system("clear");
    printf("Menu:\n\n");
    for (int i = 0; i < static_cast<int>(options_str.size()); ++i)
        printf("(%d) %s\n", i, options_str[i].c_str());
    printf("\nChoose one of the options above: ");
}

bool switch_options() {

    int op;
    scanf("%d",& op);
    op = std::min((int) options_func.size() - 1, std::abs(op));

    return (options_func[op])();
}

bool quit() { return false; }

bool create_net() {

    system("clear");
    printf("Type the number of steps: ");
    int n;
    scanf("%d",& n);
    net = Network();

    printf("\n\nSuccess!");
    wait();
    return true;
}

bool create_net_from_file() {

    system("clear");
    printf("Create network from file:\n\n");
    printf("Type the file name: ");
    char name[20];
    scanf("%s", name);
    if(strlen(name) <= 1) strcpy(name, "net");
    net = Network(name);

    printf("\n\nSuccess!");
    wait();
    return true;
}

bool show_description() {

    system("clear");
    printf("%s", net.friendly_description().c_str());
    wait();
    return true;
}

bool show_kernel() {

    system("clear");
    printf("Show convolutional kernel\n\n");
    printf("Type the layer and the order of the kernel: ");
    int l, o;
    scanf("%d %d",& l,& o);

    if (net.has_kernel(l, o)) print(net.get_kernel(l, o));
    else printf("\n\nThere is no such kernel!\n");
    wait();
    return true;
}

bool save_net() {

    system("clear");
    printf("Save network\n\n");
    printf("Type the file name: ");
    
    char name[20];
    getchar();
    scanf("%19[^\n]", name);
    if(strlen(name) < 1) strcpy(name, "net");
    printf("Name = %s;\n", name);
    net.save(name);
    printf("\n\nSuccess!");
    wait();
    return true;
}

bool train_net() {

    system("clear");
    printf("Type the file name for the train file: ");
    std::string name;
    std::cin >> name;
    if(name.size() < 1) name = "train";

    printf("Name = %s;\n", name.c_str());

    int n_imgs;
    printf("Type the number of train samples: ");
    scanf("%d",& n_imgs);
    printf("Training with %d samples;\n", n_imgs);

    net.unsupervised_learning(train_dataset, train_labels, name, n_imgs);
    printf("\n\nSuccess!");
    wait();
    return true;
}

bool test_net() {

    system("clear");
    std::string train_name, test_name;
    int train_n_imgs, test_n_imgs;

    printf("Type the name for the train file: ");
    std::cin >> train_name;
    printf("Train Name = %s;\n", train_name.c_str());
    printf("Type the number of train samples: ");
    scanf("%d",& train_n_imgs);
    printf("Training with %d samples;\n", train_n_imgs);

    printf("Type the file name for the test file: ");
    std::cin >> test_name;
    printf("Test Name = %s;\n", test_name.c_str());
    printf("Type the number of test samples: ");
    scanf("%d",& test_n_imgs);
    printf("Testing with %d samples;\n", test_n_imgs);

    net.unsupervised_testing(train_dataset, train_labels, test_dataset, test_labels,
                             train_name, test_name, train_n_imgs, test_n_imgs);
    printf("\n\nSuccess!");
    wait();
    return true;
}

bool show_layer_potentials() {

    system("clear");
    printf("Show Layer Potentials");
    printf("\n\nType the number of the layer: ");
    int l;
    scanf("%d",& l);
    if (net.has_layer(l)) print(net.get_potentials(l));
    else printf("\n\nThere is no such layer!\n");
    wait();
    return true;
}

bool wrong_op() {

    printf("\nChoose a valid option.\n");
    wait();
    return true;
}