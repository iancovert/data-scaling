'''
Preprocess datasets for experiments.
'''

import os
import torch
import argparse
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from tqdm import tqdm
from itertools import chain
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.models.resnet import ResNet50_Weights, resnet50
from opendataval.dataloader import DataFetcher
from opendataval.dataloader.register import cache
from opendataval.dataloader.util import ListDataset, CatDataset
from transformers import DistilBertModel, DistilBertTokenizerFast


# Set up argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str,
                    choices=['MiniBooNE', 'adult', 'wave_energy', 'cifar10', 'bbc', 'imdb', 'lm_reg', 'lm_clf'])
args = parser.parse_args()

# Parse arguments.
dataset = args.dataset

# Create data directory (has no effect if it already exists).
os.makedirs(f'data_files/{dataset}', exist_ok=True)
os.makedirs(f'model_results/{dataset}', exist_ok=True)


if dataset in ('imdb', 'bbc'):
    # Load raw dataset.
    if dataset == 'imdb':
        github_url = (
            'https://raw.githubusercontent.com/'
            'Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
        )
        filepath = cache(github_url, 'data_files/imdb', 'imdb.csv')
        df = pd.read_csv(filepath)
        label_dict = {'negative': 0, 'positive': 1}
        labels = torch.tensor([label_dict[label] for label in df['sentiment']])
        text_inputs = ListDataset(df['review'].values)

    elif dataset == 'bbc':
        github_url = (
            'https://raw.githubusercontent.com/'
            'mdsohaib/BBC-News-Classification/master/bbc-text.csv'
        )
        filepath = cache(github_url, 'data_files/bbc', 'bbc-text.csv')
        df = pd.read_csv(filepath)
        label_dict = {
            'business': 0,
            'entertainment': 1,
            'sport': 2,
            'tech': 3,
            'politics': 4
        }
        labels = torch.tensor([label_dict[label] for label in df['category']])
        text_inputs = ListDataset(df['text'].values)

    else:
        raise ValueError(f'Unknown dataset {dataset}')

    # Prepare tokenizer and embedding model.
    BERT_PRETRAINED_NAME = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PRETRAINED_NAME)
    bert_model = DistilBertModel.from_pretrained(BERT_PRETRAINED_NAME).cuda()

    # Embed dataset in batches.
    doc_embeddings_list = []
    dataloader = DataLoader(text_inputs, batch_size=128, pin_memory=True, num_workers=0)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Pool non-padding tokens to get document embeddings.
            bert_inputs = tokenizer(batch, padding='max_length', truncation=True, return_tensors='pt').to('cuda')
            word_embeddings = bert_model(**bert_inputs)[0]
            doc_embeddings = (
                torch.sum(word_embeddings * bert_inputs.attention_mask.unsqueeze(2), dim=1)
                / bert_inputs.attention_mask.sum(dim=1).unsqueeze(1)).cpu()
            doc_embeddings_list.append(doc_embeddings)

    # Concatenate and reduce dimensionality with PCA.
    doc_embeddings = torch.cat(doc_embeddings_list, dim=0)
    pca = PCA(n_components=256)
    doc_embeddings_pca = torch.tensor(pca.fit_transform(doc_embeddings))

    # Split the data into train, val, and test sets.
    num_points = len(labels)
    train_inds = np.arange(0, int(num_points * 0.5))
    val_inds = np.arange(int(num_points * 0.5), int(num_points * 0.75))
    test_inds = np.arange(int(num_points * 0.75), num_points)
    train_embeddings, train_labels = doc_embeddings_pca[train_inds], labels[train_inds]
    val_embeddings, val_labels = doc_embeddings_pca[val_inds], labels[val_inds]
    test_embeddings, test_labels = doc_embeddings_pca[test_inds], labels[test_inds]

    # Create datasets.
    train_dataset = CatDataset(ListDataset(text_inputs.data[train_inds]), TensorDataset(train_labels))
    val_dataset = CatDataset(ListDataset(text_inputs.data[val_inds]), TensorDataset(val_labels))
    test_dataset = CatDataset(ListDataset(text_inputs.data[test_inds]), TensorDataset(test_labels))

    print(f'Data splits: train = {len(train_dataset)}, val = {len(val_dataset)}, test = {len(test_dataset)}')
    print(f'Original embedding shape: {doc_embeddings.shape}')
    print(f'PCA embedding shape: {doc_embeddings_pca.shape}')

    data_dict = {
        # Tensors.
        'x_train': train_embeddings,
        'y_train': train_labels,
        'x_val': val_embeddings,
        'y_val': val_labels,
        'x_test': test_embeddings,
        'y_test': test_labels,

        # Datasets.
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
    torch.save(data_dict, f'data_files/{dataset}/processed.pt')


if dataset == 'cifar10':
    # Prepare data for ResNet50 feature extraction.
    image_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = CIFAR10(root='data_files/cifar10', transform=image_transforms, train=True, download=True)
    val_dataset = CIFAR10(root='data_files/cifar10', transform=image_transforms, train=False, download=True)
    test_dataset = CIFAR10(root='data_files/cifar10', transform=image_transforms, train=False, download=True)

    # Split smaller dataset into validation and test.
    val_dataset.data = val_dataset.data[:1000]
    val_dataset.targets = val_dataset.targets[:1000]
    test_dataset.data = test_dataset.data[1000:]
    test_dataset.targets = test_dataset.targets[1000:]

    # Generate embeddings.
    classifier = resnet50(weights=ResNet50_Weights.DEFAULT).eval().cuda()
    classifier.fc = torch.nn.Identity()
    train_loader = DataLoader(train_dataset, batch_size=256, pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, pin_memory=True, num_workers=0)
    embeddings_list = []
    with torch.no_grad():
        for batch in tqdm(chain(train_loader, val_loader, test_loader)):
            embed = classifier(batch[0].cuda()).cpu()
            embeddings_list.append(embed)

    # Concatenate and reduce dimensionality with PCA.
    embeddings = torch.cat(embeddings_list, dim=0)
    x_train_embed = embeddings[:len(train_dataset)]
    x_val_embed = embeddings[len(train_dataset):len(train_dataset) + len(val_dataset)]
    x_test_embed = embeddings[len(train_dataset) + len(val_dataset):]
    pca = PCA(n_components=256).fit(x_train_embed)
    x_train_embed_pca = torch.tensor(pca.transform(x_train_embed))
    x_val_embed_pca = torch.tensor(pca.transform(x_val_embed))
    x_test_embed_pca = torch.tensor(pca.transform(x_test_embed))

    # Set transforms for CIFAR ResNet.
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset.transform = image_transforms
    val_dataset.transform = image_transforms
    test_dataset.transform = image_transforms

    print(f'Data splits: train = {len(train_dataset)}, val = {len(val_dataset)}, test = {len(test_dataset)}')
    print(f'Original embedding shape: {x_train_embed.shape}')
    print(f'PCA embedding shape: {x_train_embed_pca.shape}')

    # Save dataset.
    data_dict = {
        # Tensors.
        'x_train': x_train_embed_pca,
        'y_train': torch.tensor(train_dataset.targets),
        'x_val': x_val_embed_pca,
        'y_val': torch.tensor(val_dataset.targets),
        'x_test': x_test_embed_pca,
        'y_test': torch.tensor(test_dataset.targets),

        # Datasets.
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
    torch.save(data_dict, 'data_files/cifar10/processed.pt')


if dataset in ('adult', 'MiniBooNE', 'wave_energy'):
    # Load using OpenDataVal fetcher.
    total_count_dict = {'adult': 48842, 'MiniBooNE': 72998, 'wave_energy': 72000}
    val_count = 1000
    test_count = 1000
    train_count = total_count_dict[dataset] - val_count - test_count

    # Set up fetcher.
    fetcher = DataFetcher.setup(
        dataset_name=dataset,
        cache_dir='data_files',
        force_download=False,
        random_state=np.random.RandomState(0),
        train_count=train_count,
        valid_count=val_count,
        test_count=test_count
    )

    # Extract splits.
    x_train, y_train = torch.tensor(fetcher.x_train).float(), torch.tensor(fetcher.y_train)
    x_val, y_val = torch.tensor(fetcher.x_valid).float(), torch.tensor(fetcher.y_valid)
    x_test, y_test = torch.tensor(fetcher.x_test).float(), torch.tensor(fetcher.y_test)

    if dataset == 'wave_energy':
        # Regression task, convert labels to float.
        y_train = y_train.float()
        y_val = y_val.float()
        y_test = y_test.float()
    else:
        # Classification task, convert labels to one-hot.
        y_train = torch.argmax(y_train, dim=1)
        y_val = torch.argmax(y_val, dim=1)
        y_test = torch.argmax(y_test, dim=1)

    # Create datasets.
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    print(f'Data splits: train = {len(train_dataset)}, val = {len(val_dataset)}, test = {len(test_dataset)}')
    print(f'Input shape: {x_train.shape}')

    # Save dataset.
    data_dict = {
        # Tensors.
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,

        # Datasets.
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
    torch.save(data_dict, f'data_files/{dataset}/processed.pt')


if dataset == 'lm_reg':
    # Setup.
    n = 12000
    d = 50
    noise = 0.1

    # Generate input features.
    torch.manual_seed(42)
    x = torch.randn((n, d))

    # Generate reponse variable using linear model.
    coef = torch.randn((d,))
    y = torch.matmul(x, coef) + noise * torch.randn((n,))

    # Split into train, validation, and test sets
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=2000)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=1000)

    # Create datasets.
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    print(f'Data splits: train = {len(train_dataset)}, val = {len(val_dataset)}, test = {len(test_dataset)}')
    print(f'Input shape: {x_train.shape}')

    data_dict = {
        # Tensors.
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,

        # Datasets.
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
    torch.save(data_dict, 'data_files/lm_reg/processed.pt')


if dataset == 'lm_clf':
    # Setup.
    n = 12000
    d = 50
    noise = 0.1

    # Generate input features.
    torch.manual_seed(42)
    x = torch.randn((n, d))

    # Generate response variable using logistic regression model.
    coef = torch.randn((d,))
    y = torch.matmul(x, coef) + noise * torch.randn((n,))
    y = (y > 0).long()

    # Split into train, validation, and test sets
    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=2000)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=1000)

    # Create datasets.
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    print(f'Data splits: train = {len(train_dataset)}, val = {len(val_dataset)}, test = {len(test_dataset)}')
    print(f'Input shape: {x_train.shape}')

    data_dict = {
        # Tensors.
        'x_train': x_train,
        'y_train': y_train,
        'x_val': x_val,
        'y_val': y_val,
        'x_test': x_test,
        'y_test': y_test,

        # Datasets.
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset
    }
    torch.save(data_dict, 'data_files/lm_clf/processed.pt')
