python train.py -mode translate -bert_data_path ../bert_data/cnndm -test_from ../models//bert_classifier/cnndm_bertsum_classifier_best.pt  -batch_size 30000  -log_file LOG_FILE  -test_all -block_trigram true -src_text src_text.txt -top_n_sentences 2
#python train.py -mode translate -bert_data_path ../bert_data/cnndm -test_from ../models//bert_classifier/cnndm_bertsum_classifier_best.pt  -batch_size 30000  -log_file LOG_FILE  -test_all -block_trigram true

