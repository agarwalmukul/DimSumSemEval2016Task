import pyvw

from collections import defaultdict
from nltk.corpus import wordnet as wn
import nltk



valid_labels = {
'n.act': 0,
'n.animal': 1,
'n.artifact': 2,
'n.attribute': 3,
'n.body': 4,
'n.cognition': 5,
'n.communication': 6,
'n.event': 7,
'n.feeling': 8,
'n.food': 9,
'n.group': 10,
'n.location': 11,
'n.motive': 12,
'n.natural_object': 13,
'n.other': 14,
'n.person': 15,
'n.phenomenon': 16,
'n.plant': 17,
'n.possession': 18,
'n.process': 19,
'n.quantity': 20,
'n.relation': 21,
'n.shape': 22,
'n.state': 23,
'n.substance': 24,
'n.time': 25,
'v.body': 26,
'v.change': 27,
'v.cognition': 28,
'v.communication': 29,
'v.competition': 30,
'v.consumption': 31,
'v.contact': 32,
'v.creation': 33,
'v.emotion': 34,
'v.motion': 35,
'v.perception': 36,
'v.possession': 37,
'v.social': 38,
'v.stative': 39,
'v.weather': 40}
valid_labels_rev = { v:k for k,v in valid_labels.iteritems() }


class BIO:
    # construct a BIO object using a bio type ('O', 'B' or 'I') and a
    # optionally a label (that can be used to capture the supersense tag).
    # this additionally computes a numeric_label to be used by vw
    def __init__(self, bio, label=None, pos=None, word=None):
        bio_valid = ['B']
        if bio != 'O' and bio != 'B' and bio != 'I' and bio != 'b' and bio != 'o' and bio != 'i':
            raise TypeError
        self.bio = bio
        self.label = label          # the label will only be needed for supersenses
        self.numeric_label = 1
        if self.label == None or self.label=='':
            if self.bio == 'B':
                self.numeric_label = 2
            elif self.bio == 'I':
                self.numeric_label = 3
            elif self.bio =='b':
                self.numeric_label = 4
            elif self.bio == 'i':
                self.numeric_label = 5
            elif self.bio == 'o':
                self.numeric_label = 6
        else:    #adding supersenses and supersense bio combos
            if self.bio == 'O':
                #print 'key'
                #print self.label
                self.numeric_label = int(valid_labels[self.label]) + 100
            if self.bio == 'B':
                self.numeric_label = valid_labels[self.label] + 200
            elif self.bio == 'I':
                self.numeric_label = valid_labels[self.label] + 300
            elif self.bio == 'o':
                self.numeric_label = valid_labels[self.label] + 400
            elif self.bio == 'b':
                self.numeric_label = valid_labels[self.label] + 500
            else:
                self.numeric_label = valid_labels[self.label] + 600


    # a.can_follow(b) returns true if:
    #    a is O and b is I or O or
    #    a is B and b is I or O or
    #    ...
    def can_follow(self, prev):
        cond =  (
           (self.bio == 'O' and (prev.bio == 'I' or prev.bio == 'O') ) or \
           (self.bio == 'B' and (prev.bio == 'I' or prev.bio == 'O') ) or \
           (self.bio == 'I' and (prev.bio == 'B' or prev.bio == 'I') ) )
        return cond

    # given a label, produce a list of all valid BIO items that can
    # come next.
    def valid_next(self, task=None):
        valid = []

        #if task == "MWE":

        if task == "MWE-SS":
            if self.bio == 'B':
              valid.append(BIO('I', label=None))
            if self.bio == 'I':
              valid.append(BIO('O', label=None))
              valid.append(BIO('I', label=None))
              valid.append(BIO('B', label=None))  #TODO: is this true or must mwe be a supersense??
              for i in valid_labels:
                valid.append(BIO('B', label=i))
                valid.append(BIO('O', label=i))
            if self.bio == 'O':
              valid.append(BIO('O', label=None))
              valid.append(BIO('B', label=None))  #TODO: is this true or must mwe be a supersense??
              for i in valid_labels:
                valid.append(BIO('B', label=i))
                valid.append(BIO('O', label=i))

        if task == "MWE-GAPPY":
            if self.bio == 'B':
              valid.append(BIO('I', label=None))
            if self.bio == 'I':
              valid.append(BIO('O', label=None))
              valid.append(BIO('I', label=None))
              valid.append(BIO('B', label=None))  #TODO: is this true or must mwe be a supersense??
              for i in valid_labels:
                valid.append(BIO('B', label=i))
                valid.append(BIO('O', label=i))
            if self.bio == 'O':
              valid.append(BIO('O', label=None))
              valid.append(BIO('B', label=None))  #TODO: is this true or must mwe be a supersense??
              for i in valid_labels:
                valid.append(BIO('B', label=i))
                valid.append(BIO('O', label=i))
            if self.bio == 'b':
                valid.append(BIO('i', label=None))
            if self.bio == 'i':
                valid.append(BIO('I', label=None))
                valid.append(BIO('o', label=None))
                valid.append(BIO('i', label=None))
                for i in valid_labels:
                  valid.append(BIO('o', label=i))
            if self.bio == 'o':
                valid.append(BIO('I', label=None))
                valid.append(BIO('o', label=None))
                valid.append(BIO('i', label=None))
                valid.append(BIO('b', label=None))
                for i in valid_labels:
                  valid.append(BIO('b', label=i))
                  valid.append(BIO('o', label=i))

        return valid

    # produce a human-readable string
    def __str__( self): return self.bio #return 'O' if self.bio == 'O' else (self.bio + '-' + self.label)
    def __repr__(self): return self.__str__()


    # compute equality
    def __eq__(self, other):
        if not isinstance(other, BIO): return False
        return self.bio == other.bio and self.label == other.label
    def __ne__(self, other): return not self.__eq__(other)

# convert a numerical prediction back to a BIO label
def numeric_label_to_BIO(num):
    if not isinstance(num, int):
        raise TypeError
    if num < 100:
        if num == 1:
            return BIO('O')
        elif num == 2:
            return BIO('B')
        elif num == 3:
            return BIO('I')
        elif num == 4:
            return BIO('b')
        elif num == 5:
            return BIO('i')
        elif num == 6:
            return BIO('o')
    elif num < 200: # O
        ss = valid_labels_rev[num - 100]
        return BIO('O', ss)
    elif num < 300: # B
        ss = valid_labels_rev[num - 200]
        return BIO('B', ss)
    elif num < 400:  # I
        ss = valid_labels_rev[num - 300]
        return BIO('I', ss)
    elif num < 500: # b
        ss = valid_labels_rev[num - 400]
        return BIO('b', ss)
    elif num < 600:  # i
        ss = valid_labels_rev[num - 500]
        return BIO('i', ss)
    else: # o
        ss = valid_labels_rev[num - 600]
        return BIO('o', ss)


# given a previous PREDICTED label (prev), which may be incorrect; and
# the current TRUE label (truth), generate a list of valid reference
# actions. the return type should be [BIO]. for example, if the truth
# is O or B, then regardless of what prev is the correct thing to do
# is [truth]. the most important thing is to handle the case when, for
# instance, truth is I but prev is neither I nor B
def compute_reference(prev, truth, task=None):
    ref = []
    if task == "MWE":
        if (truth.bio == 'I' and prev.bio == 'O'):
          ref = [prev]
        else:
          ref = [truth]

    elif task == "MWE-SS":

        if (truth.bio == 'O' or truth.bio == 'B'):
            ref.append(BIO(truth.bio, label=truth.label))
        elif (truth.bio == 'I' and prev.bio == 'O'):
            ref.append(BIO('O', label=None))
        else:
            ref.append(BIO(truth.bio, label=None))

    elif task == "MWE-GAPPY":

        # TODO: make decision on how to handle wrong predictions...
        if (truth.bio == 'I' and prev.bio == 'O'):      # OI
            ref.append(BIO('O', label=None))

        elif (truth.bio == 'B' and prev.bio == 'b'):    # bB
            ref.append(BIO(truth.bio, label=truth.label))

        elif (truth.bio == 'B' and prev.bio == 'o'):    # oB
            ref.append(BIO(truth.bio, label=truth.label))

        elif (truth.bio == 'B' and prev.bio == 'i'):    # iB
            ref.append(BIO(truth.bio, label=truth.label))

        elif (truth.bio == 'I' and prev.bio == 'b'):    # bI
            ref.append(BIO(truth.bio, label=None))

        elif (truth.bio == 'O' and prev.bio == 'B'):    # BO
            ref.append(BIO(truth.bio, label=truth.label))

        elif (truth.bio == 'O' and prev.bio == 'b'):
            ref.append(BIO(truth.bio, label=truth.label))

        elif (truth.bio == 'O' and prev.bio == 'i'):
            ref.append(BIO(truth.bio, label=truth.label))

        elif (truth.bio == 'O' and prev.bio == 'o'):
            ref.append(BIO(truth.bio, label=truth.label))

        elif (truth.bio == 'b' and prev.bio == 'B'):
            ref.append(BIO(truth.bio, label=truth.label))

        elif (truth.bio == 'b' and prev.bio == 'O'):
            ref.append(BIO(truth.bio, label=truth.label))

        elif (truth.bio == 'i' and prev.bio == 'O'):
            ref.append(BIO(truth.bio, label=None))

        elif (truth.bio == 'i' and prev.bio == 'B'):
            ref.append(BIO(truth.bio, label=None))

        elif (truth.bio == 'i' and prev.bio == 'I'):
            ref.append(BIO(truth.bio, label=None))

        elif (truth.bio == 'o' and prev.bio == 'O'):
            ref.append(BIO(truth.bio, label=truth.label))

        # assuming all invalid prev. pred are handled above...
        elif (truth.bio == 'O' or truth.bio == 'B' or truth.bio == 'o' or truth.bio == 'b'):
            ref.append(BIO(truth.bio, label=truth.label))
        elif (truth.bio == 'I'):
            ref.append(BIO(truth.bio, label=None))

    return ref



class MWE(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        # you must must must initialize the parent class
        # this will automatically store self.sch <- sch, self.vw <- vw
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.AUTO_HAMMING_LOSS)
        # sch.set_options( sch.AUTO_CONDITION_FEATURES| sch.IS_LDF )

    def _run(self, sentence):
        def f1(confusion):
           f = 0.
           #print confusion
           for label in confusion:
               #print confusion[label]
               label=confusion[label]
               prec = label["tp"]/(label["tp"] + label["fn"])   #TODO: double check this
               rec = label["tp"]/(label["tp"] + label["fp"])
               f += 2*prec*rec/(prec+rec) if prec+rec>0 else float('nan')
           return f


        output = []
        loss = 0.
        confusion = defaultdict(lambda:None,{})
        #print confusion
        prev = BIO('O')   # store the previous prediction
        for n in range(len(sentence)):
            with self.make_example(sentence, n) as ex:  # construct the VW example
                label,w,lemma,pos = sentence[n]
                # first, compute the numeric labels for all valid reference actions
                refs  = [ bio.numeric_label for bio in compute_reference(prev, label, task="MWE-SS") ]
                # next, because some actions are invalid based on the
                # previous decision, we need to compute a list of
                # valid actions available at this point
                valid = [ bio.numeric_label for bio in prev.valid_next(task="MWE-SS") ]
                # make a prediction
                pred  = self.sch.predict(examples   = ex,
                                         my_tag     = n+1,
                                         oracle     = refs,
                                         condition  = [(n, 'p'), (n-1, 'q')],
                                         allowed    = valid)

                # map that prediction back to a BIO label
                this = numeric_label_to_BIO(pred)

                if confusion['pred'] == None:
                  confusion['pred'] = {"tp": 0., "fp": 0., "fn": 0.}

                if confusion['valid'] == None:
                  confusion['valid'] = {"tp": 0., "fp": 0., "fn": 0.}

                # keeping track of tp, fp, and fn for sentence...
                if valid == pred:
                    confusion['pred']["tp"] += 1.
                else:
                    confusion['pred']["fp"] += 1.
                    confusion['valid']["fn"] += 1.

                # append it to output
                output.append(this)

                # update the 'previous' prediction to the current
                prev  = this

        # calculating joint f-measure
        #loss = 1. - f1(confusion)
        #self.sch.loss(loss)

        # return the list of predictions as BIO labels
        return output



    # TODO: add features here
    def make_example(self, sentence, n):
        label,w,lemma,pos = sentence[n]
        #print sentence[n]
        #offset,w,lemma,pos,mwe,parent,strength,ssense,sid = sentence[n]
        feats = {
            'w': [w],
            'l': [lemma],
            'p': [pos]
        }

        tmp = '<s>'
        if str(w[0]).isupper(): caps = "t"
        else: caps = "f"
        if str(w).isdigit():digit = "t"
        else: digit = "f"
        #print w
        #print n
        #print 'n-2'
        #print n-2
        w_p2 = sentence[n-2][0] if (n-2) >= 0 else tmp
        pos_p2 = sentence[n-2][2] if (n-2) >= 0 else tmp
        w_p1 = sentence[n-1][0] if (n-1) >= 0 else tmp
        pos_p1 = sentence[n-1][2] if (n-1) >= 0 else tmp
        w_n1 = sentence[n+1][0] if (n+1) < len(sentence) else tmp
        pos_n1 = sentence[n+1][2] if (n+1) < len(sentence) else tmp
        caps_n1 = "t" if ((n+1)<len(sentence) and str(sentence[n+1][0]).isupper()) else "f"
        #print n+2
        #print len(sentence)
        #print sentence[n+2][0]
        w_n2 = sentence[n+2][0] if (n+2) < len(sentence) else tmp
        pos_n2 = sentence[n+2][2] if (n+2) < len(sentence) else tmp


        # wordnet features:  the supersense category of the first WordNet sense of the current word. (WordNet senses are ordered roughly by frequency.)

        # suffixes: ing

        # has-supersense

        # listed as mwe in list?


        feats['a']= [pos + '_' + pos_n1] # 'p_p+1'
        feats['b']= [str(pos_p1) + '_' + pos] # 'p-1_p'
        feats['c']= [str(pos_p1) + '_' + pos + '_' + pos_n1] # 'p-1_p_p+1'
        feats['d']= [w + '_' + str(w_n1)]  # 'w_w+1'
        feats['e']= [str(w_p1) + '_' + w]  # 'w-1_w'
        feats['f']= [str(w_p1) + '_' + w + '_' + str(w_n1)]  # 'w-1_w_w+1'
        feats['g']= [caps]
        feats['h']= [caps + '_' + caps_n1]  #'cap_cap+1'
        feats['j']= [digit]

        return self.example(feats)



def make_data(BIO,filename):
    data = []
    sentence = []
    f = open(filename,'r')
    for l in f:
        l = l.strip()
        # at end of sentence
        if l == "":
            data.append(sentence)
            sentence = []
        else:
            [offset,word,lemma,pos,mwe,parent,strength,ssense,sid] = l.split('\t')
            sentence.append((BIO(mwe,ssense),word,lemma,pos))
    return data



if __name__ == "__main__":
    # input/output files
    trainfilename='dimsum16.p3.train.contiguous'
    testfilename='dimsum16.p3.test.contiguous'
    outfilename='dimsum16.p3.test.contiguous.out'

    # read in some examples to be used as training/dev set
    train_data = make_data(BIO,trainfilename)

    # initialize VW and sequence labeler as learning to search
    vw = pyvw.vw(search=3, quiet=True, search_task='hook', ring_size=1024, \
                 search_rollin='learn', search_rollout='none')

    # TODO: For ldf version ....
    # vw = pyvw.vw(search=0, csoaa_ldf=m, quiet=True, search_task='hook', ring_size=1024, \
 #               search_rollin='learn', search_rollout='ref')


    # tell VW to construct your search task object
    sequenceLabeler = vw.init_search_task(MWE)

    # train!
    # we make 5 passes over the training data, training on the first 80%
    # examples (we retain the last 20% as development data)
    print 'training!'
    N = int(0.8 * len(train_data))
    for i in xrange(5):
        print 'iteration ', i, ' ...'
        sequenceLabeler.learn(train_data[0:N])

    # now see the predictions on 20% held-out sentences
    print 'predicting!'
    hamming_loss, total_words = 0,0
    for n in range(N, len(train_data)):
        truth = [label for label,word,lemma,pos in train_data[n]]
        pred  = sequenceLabeler.predict( [(BIO('O'),word,lemma,pos) for label,word,lemma,pos in train_data[n]] )
        for i,t in enumerate(truth):
            if t != pred[i]:
                hamming_loss += 1
            total_words += 1
    #    print 'predicted:', '\t'.join(map(str, pred))
    #    print '    truth:', '\t'.join(map(str, truth))
    #    print ''
    print 'total hamming loss on dev set:', hamming_loss, '/', total_words

    # In Part II, you will have to output predictions on the test set.
    #test_data = make_data(BIO,testfilename)
    #for n in range(N, len(test_data)):
        # make predictions for current sentence
        #pred  = sequenceLabeler.predict( [(BIO('O'),word,lemma,pos) for label,word,lemma,pos in train_data[n]] )


