def split_sequences(sequences, n_steps_in, n_steps_out,n_features,X,y):
    
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        # it is a regression algorithm the inputs and outputs are the same
        seq_x, seq_y = sequences[i:end_ix, 0:n_features], sequences[end_ix:out_end_ix, 0:n_features]
        X.append(seq_x)
        y.append(seq_y)


