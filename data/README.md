**	Data
-----------------------------------------------------------


This directory includes the balanced and imbalanced folders from five carefully chosen datasets from organisms, including Caenorhabditis elegans (c_elegans), Arabidopsis thaliana (at), Drosophila melanogaster (d_mel), and Homo sapiens (hs). The directory is made up of the selected organism folder with files made up of one-hot encoded genomic sequence datasets. <br />
Adenine (A) is represented by [1 0 0 0], Cytosine (C) by [0 1 0 0], Guanine (G) by [0 0 1 0], and Thymine (T) by [0 0 0 1], with 1 designating the position of each nucleotide in the vector set. <br /> <br />

For Example: <br />
	* `all_acceptor_at` is the whole dataset, `test_acceptor_at` is the test dataset, `train_acceptor_at` is the train dataset. <br />
	* `all_acceptor_at_lbl` is the whole dataset label, `test_acceptor_at_lbl` is the test dataset label, `train_acceptor_at_lbl` is the train dataset label. <br /> <br />

This data distribution is the same for all the selected datasets organism.
