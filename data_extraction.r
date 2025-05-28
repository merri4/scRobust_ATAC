library(zellkonverter)
library(Seurat)
library(Matrix)
library(SummarizedExperiment)

# 데이터 불러오기
scatac = readH5AD("full_atlas_atac.h5ad")


# assay 분리
mtx_cnt = assay(scatac)


# histogram 그려보기
peak_values <- mtx_cnt@x
hist(peak_values, breaks = 10, main = "Histogram of Peak Values", xlab = "counts", col = "steelblue", border = "white", xlim = c(-1, 5))


# peakname를 추출/저장하기
peak_names = row.names(scatac)
special_tokens = c("PAD","SEP","UNK","CLS","MASK")

peak_names = c(special_tokens, peak_names)

df_names = data.frame(
    ID = seq_along(peak_names) - 1,
    SYMBOL = peak_names
    )
write.csv(df_names, "whole_peaks.csv", row.names = FALSE, quote = FALSE)
