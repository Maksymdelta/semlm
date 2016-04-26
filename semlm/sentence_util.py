

def print_sentence_scores(s):
    print_str = '{:8,.2f}  {:8,.2f}  {:8,.2f}  {:8,.2f}  {:5}  {}'
    print_str =  print_str.format(s.acscore,
                                  s.lmscore,
                                  s.score(lmwt=10),
                                  s.score(lmwt=10) + len(s.words) * 0.5,
                                  len(s.words),
                                  ' '.join(s.words).lower())
    print(print_str)
