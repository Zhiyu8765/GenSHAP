# Dataset lists for the LR fixed-PCA experiments

The tables below report the source datasets and the numbers of complete task
subsets actually used in the original GenSHAP LR experiment and in the three
added source-disjoint splits. The fixed preprocessing rule applies PCA with 20
components when the original feature dimension exceeds 20; datasets with at
most 20 features retain their original dimensions. In every table, an
*Unseen test* source is entirely absent from the corresponding training set.

## Original GenSHAP split (fixed PCA)

*Dataset list for the original GenSHAP LR base-model experiment under fixed
PCA. Rows marked **Unseen test** are held-out source datasets that are not used
during GenSHAP training.*

| Dataset | Use | # subsets | PCA setting |
|---|---|---:|---|
| Autos | Training | 3 | PCA (20 dimensions) |
| Bank Marketing | Training | 3 | PCA (20 dimensions) |
| Breast Cancer | Training | 2 | PCA (20 dimensions) |
| Caltech101 | Training | 3 | PCA (20 dimensions) |
| CIFAR-10 | Training | 3 | PCA (20 dimensions) |
| CIFAR-100 | Training | 3 | PCA (20 dimensions) |
| Connect-4 | Training | 3 | PCA (20 dimensions) |
| Covertype | Training | 3 | PCA (20 dimensions) |
| CovPokElec | Training | 3 | PCA (20 dimensions) |
| Credit Score | Training | 3 | PCA (20 dimensions) |
| Devnagari-Script | Training | 3 | PCA (20 dimensions) |
| Dilbert | Training | 3 | PCA (20 dimensions) |
| Dry Bean | Training | 3 | No PCA (16 dimensions) |
| EEG Eye State | Training | 3 | No PCA (14 dimensions) |
| FashionMNIST | Training | 3 | PCA (20 dimensions) |
| Gas Drift | Training | 3 | PCA (20 dimensions) |
| HAR | Training | 3 | PCA (20 dimensions) |
| hls4ml LHC Jets HLF | Training | 3 | No PCA (16 dimensions) |
| ImageNet-A | Training | 3 | PCA (20 dimensions) |
| Jannis | Training | 3 | PCA (20 dimensions) |
| Kuzushiji-MNIST | Training | 3 | PCA (20 dimensions) |
| LED(50000) | Training | 3 | PCA (20 dimensions) |
| Letter | Training | 3 | No PCA (16 dimensions) |
| MNIST | Training | 5 | PCA (20 dimensions) |
| Mozilla4 | Training | 3 | No PCA (5 dimensions) |
| Nomao | Training | 3 | PCA (20 dimensions) |
| Pendigits | Training | 3 | No PCA (16 dimensions) |
| Pokerhand | Training | 3 | No PCA (10 dimensions) |
| RandomRBF 0 0 | Training | 3 | No PCA (10 dimensions) |
| RT-IoT2022 | Training | 3 | PCA (20 dimensions) |
| SDSS17 | Training | 3 | PCA (20 dimensions) |
| Shuttle | Training | 3 | No PCA (9 dimensions) |
| Spoken Arabic Digit | Training | 3 | No PCA (14 dimensions) |
| Artificial Characters | Unseen test | 1 | No PCA (7 dimensions) |
| Iris | Unseen test | 1 | No PCA (4 dimensions) |
| STL10 | Unseen test | 1 | PCA (20 dimensions) |
| Vehicle | Unseen test | 1 | No PCA (18 dimensions) |
| Volkert | Unseen test | 2 | PCA (20 dimensions) |

## Added source-disjoint split 1 (seed 23)

*Dataset list for added LR split 1 under fixed PCA. The 80 training subsets
exclude every subset from the five source datasets marked **Unseen test**.*

| Dataset | Use | # subsets | PCA setting |
|---|---|---:|---|
| Artificial Characters | Training | 1 | No PCA (7 dimensions) |
| Bank Marketing | Training | 3 | PCA (20 dimensions) |
| Breast Cancer | Training | 2 | PCA (20 dimensions) |
| Caltech101 | Training | 2 | PCA (20 dimensions) |
| CIFAR-10 | Training | 2 | PCA (20 dimensions) |
| Connect-4 | Training | 2 | PCA (20 dimensions) |
| Covertype | Training | 3 | PCA (20 dimensions) |
| CovPokElec | Training | 3 | PCA (20 dimensions) |
| Credit Score | Training | 2 | PCA (20 dimensions) |
| Devnagari-Script | Training | 3 | PCA (20 dimensions) |
| Dilbert | Training | 2 | PCA (20 dimensions) |
| Dry Bean | Training | 3 | No PCA (16 dimensions) |
| EEG Eye State | Training | 3 | No PCA (14 dimensions) |
| FashionMNIST | Training | 3 | PCA (20 dimensions) |
| Gas Drift | Training | 3 | PCA (20 dimensions) |
| HAR | Training | 3 | PCA (20 dimensions) |
| Jannis | Training | 2 | PCA (20 dimensions) |
| Kuzushiji-MNIST | Training | 2 | PCA (20 dimensions) |
| LED(50000) | Training | 3 | PCA (20 dimensions) |
| Letter | Training | 2 | No PCA (16 dimensions) |
| MNIST | Training | 5 | PCA (20 dimensions) |
| Mozilla4 | Training | 3 | No PCA (5 dimensions) |
| Nomao | Training | 3 | PCA (20 dimensions) |
| Pendigits | Training | 3 | No PCA (16 dimensions) |
| Pokerhand | Training | 3 | No PCA (10 dimensions) |
| RandomRBF 0 0 | Training | 3 | No PCA (10 dimensions) |
| RT-IoT2022 | Training | 2 | PCA (20 dimensions) |
| SDSS17 | Training | 1 | PCA (20 dimensions) |
| Shuttle | Training | 2 | No PCA (9 dimensions) |
| Spoken Arabic Digit | Training | 3 | No PCA (14 dimensions) |
| Vehicle | Training | 1 | No PCA (18 dimensions) |
| Volkert | Training | 2 | PCA (20 dimensions) |
| Autos | Unseen test | 1 | PCA (20 dimensions) |
| CIFAR-100 | Unseen test | 1 | PCA (20 dimensions) |
| hls4ml LHC Jets HLF | Unseen test | 1 | No PCA (16 dimensions) |
| ImageNet-A | Unseen test | 1 | PCA (20 dimensions) |
| Iris | Unseen test | 1 | No PCA (4 dimensions) |

## Added source-disjoint split 2 (seed 42)

*Dataset list for added LR split 2 under fixed PCA. The 80 training subsets
exclude every subset from the five source datasets marked **Unseen test**.*

| Dataset | Use | # subsets | PCA setting |
|---|---|---:|---|
| Artificial Characters | Training | 1 | No PCA (7 dimensions) |
| Autos | Training | 3 | PCA (20 dimensions) |
| Breast Cancer | Training | 2 | PCA (20 dimensions) |
| Caltech101 | Training | 2 | PCA (20 dimensions) |
| CIFAR-10 | Training | 2 | PCA (20 dimensions) |
| CIFAR-100 | Training | 3 | PCA (20 dimensions) |
| Connect-4 | Training | 3 | PCA (20 dimensions) |
| CovPokElec | Training | 3 | PCA (20 dimensions) |
| Credit Score | Training | 3 | PCA (20 dimensions) |
| Devnagari-Script | Training | 3 | PCA (20 dimensions) |
| Dilbert | Training | 1 | PCA (20 dimensions) |
| Dry Bean | Training | 3 | No PCA (16 dimensions) |
| EEG Eye State | Training | 3 | No PCA (14 dimensions) |
| FashionMNIST | Training | 3 | PCA (20 dimensions) |
| hls4ml LHC Jets HLF | Training | 2 | No PCA (16 dimensions) |
| ImageNet-A | Training | 3 | PCA (20 dimensions) |
| Iris | Training | 1 | No PCA (4 dimensions) |
| Jannis | Training | 3 | PCA (20 dimensions) |
| LED(50000) | Training | 3 | PCA (20 dimensions) |
| Letter | Training | 3 | No PCA (16 dimensions) |
| MNIST | Training | 5 | PCA (20 dimensions) |
| Mozilla4 | Training | 2 | No PCA (5 dimensions) |
| Nomao | Training | 2 | PCA (20 dimensions) |
| Pendigits | Training | 3 | No PCA (16 dimensions) |
| Pokerhand | Training | 2 | No PCA (10 dimensions) |
| RandomRBF 0 0 | Training | 3 | No PCA (10 dimensions) |
| RT-IoT2022 | Training | 1 | PCA (20 dimensions) |
| SDSS17 | Training | 3 | PCA (20 dimensions) |
| Shuttle | Training | 3 | No PCA (9 dimensions) |
| Spoken Arabic Digit | Training | 2 | No PCA (14 dimensions) |
| STL10 | Training | 1 | PCA (20 dimensions) |
| Vehicle | Training | 1 | No PCA (18 dimensions) |
| Volkert | Training | 2 | PCA (20 dimensions) |
| Bank Marketing | Unseen test | 1 | PCA (20 dimensions) |
| Covertype | Unseen test | 1 | PCA (20 dimensions) |
| Gas Drift | Unseen test | 1 | PCA (20 dimensions) |
| HAR | Unseen test | 1 | PCA (20 dimensions) |
| Kuzushiji-MNIST | Unseen test | 1 | PCA (20 dimensions) |

## Added source-disjoint split 3 (seed 2026)

*Dataset list for added LR split 3 under fixed PCA. The 80 training subsets
exclude every subset from the five source datasets marked **Unseen test**.*

| Dataset | Use | # subsets | PCA setting |
|---|---|---:|---|
| Artificial Characters | Training | 1 | No PCA (7 dimensions) |
| Autos | Training | 3 | PCA (20 dimensions) |
| Bank Marketing | Training | 1 | PCA (20 dimensions) |
| Breast Cancer | Training | 2 | PCA (20 dimensions) |
| CIFAR-10 | Training | 2 | PCA (20 dimensions) |
| CIFAR-100 | Training | 3 | PCA (20 dimensions) |
| Connect-4 | Training | 1 | PCA (20 dimensions) |
| Covertype | Training | 3 | PCA (20 dimensions) |
| CovPokElec | Training | 3 | PCA (20 dimensions) |
| Credit Score | Training | 3 | PCA (20 dimensions) |
| Devnagari-Script | Training | 3 | PCA (20 dimensions) |
| Dilbert | Training | 3 | PCA (20 dimensions) |
| Dry Bean | Training | 3 | No PCA (16 dimensions) |
| FashionMNIST | Training | 2 | PCA (20 dimensions) |
| Gas Drift | Training | 2 | PCA (20 dimensions) |
| HAR | Training | 3 | PCA (20 dimensions) |
| hls4ml LHC Jets HLF | Training | 3 | No PCA (16 dimensions) |
| ImageNet-A | Training | 3 | PCA (20 dimensions) |
| Iris | Training | 1 | No PCA (4 dimensions) |
| Jannis | Training | 3 | PCA (20 dimensions) |
| Kuzushiji-MNIST | Training | 3 | PCA (20 dimensions) |
| LED(50000) | Training | 3 | PCA (20 dimensions) |
| Letter | Training | 3 | No PCA (16 dimensions) |
| MNIST | Training | 5 | PCA (20 dimensions) |
| Nomao | Training | 3 | PCA (20 dimensions) |
| Pendigits | Training | 2 | No PCA (16 dimensions) |
| Pokerhand | Training | 2 | No PCA (10 dimensions) |
| RandomRBF 0 0 | Training | 3 | No PCA (10 dimensions) |
| SDSS17 | Training | 2 | PCA (20 dimensions) |
| Shuttle | Training | 3 | No PCA (9 dimensions) |
| Spoken Arabic Digit | Training | 2 | No PCA (14 dimensions) |
| Vehicle | Training | 1 | No PCA (18 dimensions) |
| Caltech101 | Unseen test | 1 | PCA (20 dimensions) |
| EEG Eye State | Unseen test | 1 | No PCA (14 dimensions) |
| Mozilla4 | Unseen test | 1 | No PCA (5 dimensions) |
| RT-IoT2022 | Unseen test | 1 | PCA (20 dimensions) |
| Volkert | Unseen test | 1 | PCA (20 dimensions) |
