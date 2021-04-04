# CGanTransformers
This project was completed as part of the machine learning practical group project. The aim of this project was to investigate a novel question in machine learning and produce a report of the findings. The abstract of the report is detailed below.

Please note that the code was not written with reusability in mind and much of it is drawn from a variety of sources. An improved and more usable version may be pushed in the near future.

## Paper Abstract
In this paper we propose a novel architecture for tackling the problem of sequence to sequence translation on unpaired datasets and focus on the style transfer of Pop-Rock music into NES videogame style music. Previous works in this domain tend to revert to using image based representations of music, which we believe can obscure the discrete structure of musical events and can struggle to maintain coherence for lengthy sequences. We propose an alternative method based on CycleGANs which can incorporate a variety of sequence to sequence models in order to tackle this issue.
This would allow us to incorporate more powerful NLP models, e.g. transformers, which can better represent musical structure. We compare the performance of two models trained with our methodology; one built upon LSTMs while the other utilises transformers. The comparison is done both quantitatively and qualitatively. The quantitative analysis indicates a potential limitation in the discriminator networks which is further reinforced by the qualitative analysis.

Much of the data utilities here are pulled from <https://github.com/chrisdonahue/LakhNES>.


