# dnn-im2col

DNN framework using im2col

[![Build Status](https://travis-ci.org/hiroyam/dnn-im2col.svg?branch=master)](https://travis-ci.org/hiroyam/dnn-im2col)

--- 

#### これは何？

DNNの実装に関して実験するために書いたDNNのミニマルなフレームワークです。

im2colで畳み込みを密行列の積に変換し、GEMMで計算しているのが特徴です。

BLASをcuBLASなどに置換すると高速化できます。

[cuDNNの論文](https://arxiv.org/pdf/1410.0759.pdf)を参考にしました。
