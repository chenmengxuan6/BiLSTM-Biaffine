
### 输入数据格式请处理成BIO格式，如下：
```
IL-2	B-DNA
gene	I-DNA
expression	O
and	O
NF-kappa	B-protein
B	I-protein
activation	O
through	O
CD28	B-protein
requires	O
reactive	O
oxygen	O
production	O
by	O
5-lipoxygenase	B-protein
.	O

Activation	O
of	O
the	O
CD28	B-protein
surface	I-protein
receptor	I-protein
provides	O
a	O
major	O
costimulatory	O
signal	O
for	O
T	O
cell	O
activation	O
resulting	O
in	O
enhanced	O
production	O
of	O
interleukin-2	B-protein
(	O
IL-2	B-protein
)	O
and	O
cell	O
proliferation	O
.	O
```

### 运行的环境
```
python == 3.7.4
pytorch == 1.3.1 
pytorch-crf == 0.7.2  
pytorch-transformers == 1.2.0               
```



