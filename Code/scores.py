import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
from utils import *
from model import *

def GetArgs():
    parser = argparse.ArgumentParser(description='Uncertainity Scores')
    parser.add_argument('--bs', default='512')
    parser.add_argument("--ood_dataset",default="cifar10")
    parser.add_argument("--dropout",default=0)
    parser.add_argument("--mode",default="fine")
    args = parser.parse_args()
    return args

def GetData(args):
    cifar100_data_train = tfds.load('cifar100', split='train', shuffle_files=False,data_dir = DATA_DIR)
    cifar100_data_test = tfds.load('cifar100', split='test', shuffle_files=False,data_dir = DATA_DIR)
    in_train = prepare_pure_dataset(cifar100_data_train, 100, shuffle=False, batch_size=int(args.bs))
    in_test = prepare_pure_dataset(cifar100_data_test, 100, shuffle=False, batch_size=int(args.bs))
    if args.ood_dataset == "cifar10":
        cifar10_data_test = tfds.load('cifar10', split='test', shuffle_files=False,data_dir = DATA_DIR)
        out_test = prepare_pure_dataset(cifar10_data_test, 10, shuffle=False, batch_size=int(args.bs))
    elif args.ood_dataset == "fashion_mnist":
        fmnist_data_test = tfds.load('fashion_mnist', split='test', shuffle_files=False,data_dir = DATA_DIR)
        out_test = prepare_pure_dataset(fmnist_data_test, 10, shuffle=False, batch_size=int(args.bs))
    elif args.ood_dataset == "mixup":
        cifar10_data_test = tfds.load('cifar10', split='test', shuffle_files=False,data_dir = DATA_DIR)
        cifar_mixup = mix_up(cifar100_data_test,cifar10_data_test,alpha=0.1)
        out_test = prepare_pure_dataset(cifar_mixup, 10, shuffle=False, batch_size=int(args.bs))
    return in_train,in_test,out_test

def GetModel(args):
    #Get saved model
    num_classes = 100 if args.mode == "fine" else 20
    model = Neural_Img_Clf_model(dropout=float(args.dropout),no_classes=num_classes)
    model = torch.nn.DataParallel(model)
    model.cuda()
    checkpoint = torch.load(f'../Models/vit_cifar100_{args.mode}.t7')
    model.load_state_dict(checkpoint["model"])

    return model

def GetStandardMBDistance(in_train_embeds,in_train_labels,in_test_embeds,out_test_embeds):
    onehots, scores, description, maha_intermediate_dict = get_scores(
        np.array(in_train_embeds)[:,:],
        in_train_labels,
        np.array(in_test_embeds)[:,:],
        np.array(out_test_embeds)[:,:],
        indist_classes=100,
        subtract_mean = False,
        normalize_to_unity = False,
        subtract_train_distance = False,
    )

    class_means = maha_intermediate_dict["class_means"]
    class_cov_invs = maha_intermediate_dict["class_cov_invs"]


    indist_dists = []
    for c in range(100):
        indist_offset_now = in_test_embeds - class_means[c].reshape([1,-1])
        maha_dists_now = np.sum(np.matmul(indist_offset_now,class_cov_invs[c])*indist_offset_now,axis=1)
        indist_dists.append(maha_dists_now)

    outdist_dists = []
    for c in range(100):
        outdist_offset_now = out_test_embeds - class_means[c].reshape([1,-1])
        maha_dists_now = np.sum(np.matmul(outdist_offset_now,class_cov_invs[c])*outdist_offset_now,axis=1)
        outdist_dists.append(maha_dists_now)

    indist_dists_byclass = np.stack(indist_dists,axis=1)
    indist_min = np.min(indist_dists_byclass,axis=1)

    outdist_dists_byclass = np.stack(outdist_dists,axis=1)
    outdist_min = np.min(outdist_dists_byclass,axis=1)

    onehots = np.array([1]*len(outdist_min) + [0]*len(indist_min))
    scores = np.concatenate([outdist_min,indist_min],axis=0)

    print("CIFAR-100 score = "+str(np.mean(indist_min))+"+-"+str(np.std(indist_min)))
    print("CIFAR-10 score = "+str(np.mean(outdist_min))+"+-"+str(np.std(outdist_min)))

    auroc, to_replot_dict = get_auroc(
        onehots, 
        scores, 
        make_plot=False,
        add_to_title="ViT-L_16 on CIFAR-100 vs CIFAR-10\nStandard Mahalanobis"
        )
    print("AUROC, Standard Mahalanobis: ",auroc)

    return maha_intermediate_dict,indist_dists,outdist_dists


def GetRelativeMBDistance(in_train_embeds,in_train_labels,in_test_embeds,out_test_embeds,maha_intermediate_dict,indist_dists,outdist_dists):
    train_mean = maha_intermediate_dict["mean"]
    train_cov_inv = maha_intermediate_dict["cov_inv"]

    onehots, scores, description, maha_intermediate_dict = get_scores(
            np.array(in_train_embeds)[:,:],
            in_train_labels,
            np.array(in_test_embeds)[:,:],
            np.array(out_test_embeds)[:,:],
            indist_classes=100,
            subtract_mean = False,
            normalize_to_unity = False,
            subtract_train_distance = True,
        )


    indist_dists_byclass = np.stack(indist_dists,axis=1)
    indist_min = np.min(indist_dists_byclass,axis=1)

    outdist_dists_byclass = np.stack(outdist_dists,axis=1)
    outdist_min = np.min(outdist_dists_byclass,axis=1)

    prelogits = in_test_embeds
    offset_now = prelogits - np.array(train_mean).reshape([1,-1]).astype(np.float64)
    offset_now = offset_now.astype(np.float64)
    train_maha_dist = np.einsum("ai,ij->aj",offset_now,np.array(train_cov_inv).astype(np.float64))
    train_maha_dist = np.einsum("aj,aj->a",train_maha_dist,offset_now)
    indist_train_dist = train_maha_dist

    prelogits = out_test_embeds
    offset_now = prelogits - np.array(train_mean).reshape([1,-1]).astype(np.float64)
    offset_now = offset_now.astype(np.float64)
    train_maha_dist = np.einsum("ai,ij->aj",offset_now,np.array(train_cov_inv).astype(np.float64))
    train_maha_dist = np.einsum("aj,aj->a",train_maha_dist,offset_now)
    outdist_train_dist = train_maha_dist

    outdist_scores = outdist_min-outdist_train_dist
    indist_scores = indist_min-indist_train_dist

    onehots = np.array([1]*len(outdist_min) + [0]*len(indist_min))
    scores = np.concatenate([outdist_scores,indist_scores],axis=0)

    print("CIFAR-100 score = "+str(np.mean(indist_scores))+"+-"+str(np.std(indist_scores)))
    print("CIFAR-10 score = "+str(np.mean(outdist_scores))+"+-"+str(np.std(outdist_scores)))

    auroc, to_replot_dict = get_auroc(
        onehots, 
        scores, 
        make_plot=False,
        add_to_title="ViT-L_16 on CIFAR-100 vs CIFAR-10\nRelative Mahalanobis"
        )
    print("AUROC, Relative Mahalanobis: ",auroc)

def GetSoftmaxScores(in_test_logits,out_test_logits):
    scores = np.array(
    np.concatenate([
     np.max(np_softmax(in_test_logits),axis=-1),
     np.max(np_softmax(out_test_logits),axis=-1),
    ],axis=0)
    )

    onehots = np.array(
        [1]*len(in_test_logits)+[0]*len(out_test_logits)
    )

    auroc, to_replot_dict = get_auroc(
        onehots, 
        scores, 
        make_plot=False,
        add_to_title="ViT-L_16 on CIFAR-100 vs CIFAR-10\nMax of Softmax Probs",
        swap_classes=True,
        )

    print("AUROC, Max of Softmax Probs: ",auroc)


if __name__ == "__main__":
    DATA_DIR = "../Data/"
    args = GetArgs()
    in_train,in_test,out_test = GetData(args)
    model = GetModel(args)

    N_train = 5000
    N_test = 1000

    out_test_prelogits, out_test_logits, out_test_labels = standalone_get_prelogits(model,
    out_test,
    int(args.bs),
    image_count=N_test
    )

    in_test_prelogits, in_test_logits, in_test_labels = standalone_get_prelogits( model,
    in_test, 
    int(args.bs),
    image_count=N_test
    )

    in_train_prelogits, in_train_logits, in_train_labels = standalone_get_prelogits(model,
    in_train, 
    int(args.bs),
    image_count=N_train
    )

    finetune_test_acc = np.mean(
    np.argmax(in_test_logits,axis=-1) == in_test_labels
    )

    print("CIFAR-100 test accuracy = "+str(finetune_test_acc))

    finetune_train_acc = np.mean(
    np.argmax(in_train_logits,axis=-1) == in_train_labels
    )

    print("CIFAR-100 train accuracy = "+str(finetune_train_acc))



    maha_intermediate_dict,indist_dists,outdist_dists = GetStandardMBDistance(in_train_prelogits,in_train_labels,in_test_prelogits,out_test_prelogits)

    GetRelativeMBDistance(in_train_prelogits,in_train_labels,in_test_prelogits,out_test_prelogits,maha_intermediate_dict,indist_dists,outdist_dists)

    GetSoftmaxScores(in_test_logits,out_test_logits)













