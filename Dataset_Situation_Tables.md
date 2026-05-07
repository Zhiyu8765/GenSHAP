# Dataset Situation Tables

The tables below are the revised four-column version for the paper. The `Unseen test` rows are held out at the dataset level and are not used during GenSHAP training.

## LR base model

| Dataset               | Use         |   # subsets | PCA setting              |
|:----------------------|:------------|------------:|:-------------------------|
| Artificial Characters | Training    |           3 | PCA                      |
| Autos                 | Training    |           3 | PCA                      |
| Bank Marketing        | Training    |           3 | PCA                      |
| Caltech101            | Training    |           3 | PCA                      |
| CIFAR-10              | Training    |           3 | PCA                      |
| CIFAR-100             | Training    |           3 | PCA                      |
| Connect-4             | Training    |           3 | PCA                      |
| Covertype             | Training    |           3 | PCA                      |
| CovPokElec            | Training    |           3 | PCA                      |
| Credit Score          | Training    |           3 | PCA                      |
| Devnagari-Script      | Training    |           3 | PCA                      |
| Dilbert               | Training    |           3 | PCA                      |
| Dry Bean              | Training    |           3 | PCA                      |
| FashionMNIST          | Training    |           3 | PCA                      |
| Financial Risk        | Training    |           3 | PCA                      |
| Gas Drift             | Training    |           3 | PCA                      |
| HAR                   | Training    |           3 | PCA                      |
| hls4ml LHC Jets HLF   | Training    |           3 | PCA                      |
| ImageNet-A            | Training    |           3 | PCA                      |
| Jannis                | Training    |           3 | PCA                      |
| Kuzushiji-MNIST       | Training    |           3 | PCA                      |
| LED(50000)            | Training    |           3 | PCA                      |
| Letter                | Training    |           1 | PCA                      |
| Nomao                 | Training    |           3 | PCA                      |
| Pendigits             | Training    |           3 | PCA                      |
| Pokerhand             | Training    |           3 | PCA                      |
| RandomRBF 0 0         | Training    |           3 | PCA                      |
| RT-IoT2022            | Training    |           3 | PCA                      |
| SDSS17                | Training    |           3 | PCA                      |
| Shuttle               | Training    |           3 | PCA                      |
| Spoken Arabic Digit   | Training    |           3 | PCA                      |
| STL10                 | Training    |           3 | PCA                      |
| Volkert               | Training    |           3 | PCA                      |
| WBCAtt                | Training    |           3 | PCA                      |
| Artificial Characters | Unseen test |           1 | PCA                      |
| Iris                  | Unseen test |           1 | No PCA (dimensions < 20) |
| STL10                 | Unseen test |           1 | PCA                      |
| Vehicle               | Unseen test |           1 | No PCA (dimensions < 20) |
| Volkert               | Unseen test |           2 | PCA                      |

## SVM base model

| Dataset               | Use         |   # subsets | PCA setting              |
|:----------------------|:------------|------------:|:-------------------------|
| Artificial Characters | Training    |           3 | PCA                      |
| Autos                 | Training    |           3 | PCA                      |
| Bank Marketing        | Training    |           3 | PCA                      |
| Breast Cancer         | Training    |           1 | PCA                      |
| Caltech101            | Training    |           3 | PCA                      |
| CIFAR-10              | Training    |           3 | PCA                      |
| CIFAR-100             | Training    |           3 | PCA                      |
| Connect-4             | Training    |           3 | PCA                      |
| Covertype             | Training    |           3 | PCA                      |
| CovPokElec            | Training    |           3 | PCA                      |
| Credit Score          | Training    |           3 | PCA                      |
| Devnagari-Script      | Training    |           3 | PCA                      |
| Dilbert               | Training    |           3 | PCA                      |
| Dry Bean              | Training    |           3 | PCA                      |
| FashionMNIST          | Training    |           3 | PCA                      |
| Financial Risk        | Training    |           3 | PCA                      |
| Gas Drift             | Training    |           3 | PCA                      |
| HAR                   | Training    |           3 | PCA                      |
| hls4ml LHC Jets HLF   | Training    |           3 | PCA                      |
| ImageNet-A            | Training    |           3 | PCA                      |
| Jannis                | Training    |           3 | PCA                      |
| Kuzushiji-MNIST       | Training    |           3 | PCA                      |
| LED(50000)            | Training    |           3 | PCA                      |
| Letter                | Training    |           3 | PCA                      |
| MNIST                 | Training    |           3 | PCA                      |
| Nomao                 | Training    |           3 | PCA                      |
| Pendigits             | Training    |           3 | PCA                      |
| Pokerhand             | Training    |           3 | PCA                      |
| RandomRBF 0 0         | Training    |           3 | PCA                      |
| RT-IoT2022            | Training    |           3 | PCA                      |
| SDSS17                | Training    |           3 | PCA                      |
| Shuttle               | Training    |           3 | PCA                      |
| Spoken Arabic Digit   | Training    |           3 | PCA                      |
| WBCAtt                | Training    |           3 | PCA                      |
| Breast Cancer         | Unseen test |           2 | PCA                      |
| Iris                  | Unseen test |           1 | No PCA (dimensions < 20) |
| STL10                 | Unseen test |           3 | PCA                      |
| Vehicle               | Unseen test |           1 | No PCA (dimensions < 20) |
| Volkert               | Unseen test |           3 | PCA                      |
