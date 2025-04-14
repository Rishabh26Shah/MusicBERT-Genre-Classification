# Binarize labels
fairseq-preprocess \
  --only-source \
  --trainpref processed/midi_train.label \
  --validpref processed/midi_valid.label \
  --testpref processed/midi_test.label \
  --destdir bin_data_genre/label \
  --workers 8 \
  --srcdict processed/label/dict.txt

