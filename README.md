

We build on top of the following work:

- **[berenslab/iclr-dataset](https://github.com/berenslab/iclr-dataset)**: Provides a complete scrape of ICLR submissions from OpenReview with metadata including titles, abstracts, authors, decisions, reviewer scores, and keyword-based labels. The dataset serves as a benchmark for embedding quality evaluation.

- **[SachinKonan/AutoReviewer](https://github.com/SachinKonan/AutoReviewer)**: Extracts and processes ICLR papers and reviews from OpenReview. It downloads PDFs, converts them to markdown using MinerU, normalizes review content, and creates structured datasets for analysis.

- **[skonan/iclr-reviews-2020-2026](https://huggingface.co/datasets/skonan/iclr-reviews-2020-2026)**: A Hugging Face dataset containing ICLR papers (2020-2026) with full paper content, extracted images, reviews, meta-reviews, and metadata for comprehensive analysis.

The dataset (version 24v2) is described in [González-Márquez & Kobak, Learning representations of learning representations, DMLR workshop at ICLR 2024](https://openreview.net/forum?id=2OObXL3AaZ) ([arXiv 2404.08403](https://arxiv.org/abs/2404.08403)). Please cite as follows:

```
@inproceedings{gonzalez2024learning,
  title={Learning representations of learning representations},
  author={Gonz{\'a}lez-M{\'a}rquez, Rita and Kobak, Dmitry},
  booktitle={Data-centric Machine Learning Research (DMLR) workshop at ICLR 2024},
  year={2024}
}
```