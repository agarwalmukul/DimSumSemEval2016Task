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
    def __init__(self, bio, label=''):
        if bio != 'O' and bio != 'B' and bio != 'I':
            raise TypeError
        self.bio = bio
        self.label = label          # the label will only be needed for supersenses
        self.numeric_label = 1
        if self.label == None or self.label == '': 
            if self.bio == 'B':
                self.numeric_label = 2 
            elif self.bio == 'I':
                self.numeric_label = 3 
        else:    #adding supersenses and supersense bio combos
            if self.bio == 'O': 
                self.numeric_label = valid_labels[self.label] + 100
            if self.bio == 'B': 
                self.numeric_label = valid_labels[self.label] + 200  
            elif self.bio == 'I':
                self.numeric_label = valid_labels[self.label] + 300
            
                   
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
    def valid_next(self):    
         valid = []
         if self.bio == 'B': 
           valid.append(BIO('I'))
         if self.bio == 'I':
           valid.append(BIO('O')) 
           valid.append(BIO('I')) 
           valid.append(BIO('B'))  
           for i in valid_labels:
             valid.append(BIO('B', label=i))
             valid.append(BIO('O', label=i))
         if self.bio == 'O': 
           valid.append(BIO('O')) 
           valid.append(BIO('B'))  
           for i in valid_labels: 
             valid.append(BIO('B', label=i))
             valid.append(BIO('O', label=i)) 

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
    elif num < 200: # O
        ss = valid_labels_rev[num - 100]
        return BIO('O', ss)
    elif num < 300: # B
        ss = valid_labels_rev[num - 200]
        return BIO('B', ss)
    elif num < 400:  # I
        ss = valid_labels_rev[num - 300]
        return BIO('I', ss)


        
# given a previous PREDICTED label (prev), which may be incorrect; and
# the current TRUE label (truth), generate a list of valid reference
# actions. the return type should be [BIO]. for example, if the truth
# is O or B, then regardless of what prev is the correct thing to do
# is [truth]. the most important thing is to handle the case when, for
# instance, truth is I but prev is neither I nor B
def compute_reference(prev, truth): 
    ref = []

    if (truth.bio == 'O' or truth.bio == 'B'):
        ref.append(BIO(truth.bio, label=truth.label))
    elif (truth.bio == 'I' and prev.bio == 'O'):
        ref.append(BIO('O'))
    else:
        ref.append(BIO(truth.bio))
           
    return ref


    
class MWE(pyvw.SearchTask):
    def __init__(self, vw, sch, num_actions):
        # you must must must initialize the parent class
        # this will automatically store self.sch <- sch, self.vw <- vw
        pyvw.SearchTask.__init__(self, vw, sch, num_actions)
        sch.set_options( sch.AUTO_CONDITION_FEATURES )
        # sch.set_options( sch.AUTO_CONDITION_FEATURES| sch.IS_LDF )

    def _run(self, sentence):
        def f1(conf): 
           f = 0.
           for l in conf:
               prec = conf[l]["tp"]/(conf[l]["tp"] + conf[l]["fn"])   if conf[l]["tp"] + conf[l]["fn"]>0 else float(0.)
               rec = conf[l]["tp"]/(conf[l]["tp"] + conf[l]["fp"]) if conf[l]["tp"] + conf[l]["fp"]>0 else float(0.)
               f += 2*prec*rec/(prec+rec) if prec+rec>0 else float(0.)
           return f

        output = []
        loss = 0.
        confusion = defaultdict(lambda:None)       
        prev = BIO('O')   # store the previous prediction
        for n in range(len(sentence)):
            with self.make_example(sentence, n) as ex:  # construct the VW example
                label  = sentence[n][0]
                # first, compute the numeric labels for all valid reference actions
                
                refs  = [ bio.numeric_label for bio in compute_reference(prev, label) ]   
                
                # next, because some actions are invalid based on the
                # previous decision, we need to compute a list of
                # valid actions available at this point
                valid = [ bio.numeric_label for bio in prev.valid_next() ] 
                
                
                # make a prediction
                pred  = self.sch.predict(examples   = ex,
                                         my_tag     = n+1,
                                         oracle     = refs,
                                         condition  = [(n, 'p'), (n-1, 'q')],
                                         allowed    = valid)  
               
                # map that prediction back to a BIO label
                this = numeric_label_to_BIO(pred)
                
                if confusion[pred] == None:
                  confusion[pred] = {"tp": 0., "fp": 0., "fn": 0.}
                if refs != [] and confusion[refs[0]] == None:
                  confusion[refs[0]] = {"tp": 0., "fp": 0., "fn": 0.}
                
                # keeping track of tp, fp, and fn for sentence...
                if refs == []:
                    confusion[pred]["fp"] += 1.
                elif refs[0] == pred:
                    confusion[pred]["tp"] += 1.
                else: 
                    confusion[pred]["fp"] += 1.
                    confusion[refs[0]]["fn"] += 1.
                     
                # append it to output
                output.append(this)
                
                # update the 'previous' prediction to the current
                prev  = this
                # ex.finish()
 
 
        # calculating f-score
        f = f1(confusion)
        loss = 1. - f
        self.sch.loss(loss)
        
        # return the list of predictions as BIO labels
        return output



    # TODO: add features here 
    def make_example(self, sentence, n):
        lemma = sentence[n][2]
        pos = sentence[n][3]
        w = sentence[n][1]
      
        feats = {
             'w': [w],
             'l': [lemma],
             'p': [pos]
        }
      
        tmp = '<s>'
        if (w[0].isupper()): 
           caps = "t" 
        else: 
           caps = "f"
        
        if w.isdigit(): 
           digit = "t"  
        else: 
           digit = "f"
      
        if (n-2) >= 0: 
           w_p2 = sentence[n-2][1] 
           pos_p2 = sentence[n-2][2] 
        else: 
           w_p2  = tmp
           pos_p2 = tmp
      
        if (n-1) >= 0: 
           w_p1 = sentence[n-1][1] 
           pos_p1 = sentence[n-1][2] 
        else: 
           w_p1 = tmp
           pos_p1 = tmp
      
        if (n+1) < len(sentence): 
          w_n1 = sentence[n+1][1] 
          pos_n1 = sentence[n+1][2]
          if (sentence[n+1][1][0].isupper()): 
            caps_n1 = "t"
          else:  
            caps_n1 = "f"
        else: 
          w_n1 = tmp
          pos_n1 = tmp
          caps_n1 = "f"
        
        if (n+2) < len(sentence): 
          w_n2 = sentence[n+2][1]  
          pos_n2 = sentence[n+2][2] 
        else: 
          w_n2 = tmp
          pos_n2 = tmp
      

        feats['a'] = [pos + '_' + pos_n1] # 'p_p+1'
        feats['b'] = [pos_p1 + '_' + pos] # 'p-1_p'
        feats['c'] = [pos_p1 + '_' + pos + '_' + pos_n1] # 'p-1_p_p+1'
        feats['d'] = [w + '_' + w_n1]  # 'w_w+1'
        feats['e'] = [w_p1 + '_' + w]  # 'w-1_w'
        feats['f'] = [w_p1 + '_' + w + '_' + w_n1]  # 'w-1_w_w+1'
        feats['g'] = [caps]
        feats['h'] = [caps + '_' + caps_n1]  #'cap_cap+1'
        feats['j'] = [digit]
      
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
            sentence.append((BIO(mwe, label=ssense),word,lemma,pos))
    return data


def make_test_data(BIO,filename):
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
            [offset,word,lemma,pos] = l.split('\t')
            sentence.append(offset,word,lemma,pos,'','','','','')
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
        
        
        def print_joint(bio): 
         if None: 
           "I'm none + " + bio.label
         else: 
          print bio.bio + '-' + bio.label
          bio.bio + '-' + bio.label
        
        for i in len(train_data[n]): 
           print train_data[n][i]
        
        print 'predicted:', '\t'.join(map(print_joint, pred))
        # print '    truth:', '\t'.join(map(print_joint, truth))

        
        for i,t in enumerate(truth):
            print pred.__class__.__name__
            print i
            print pred[i]
            print t

            if t != pred[i]:
                hamming_loss += 1
            total_words += 1
    
    print 'total hamming loss on dev set:', hamming_loss, '/', total_words

    # In Part II, you will have to output predictions on the test set.
    # test_data = make_test_data(BIO,testfilename)
    # for n in range(N, len(test_data)):
    #     #make predictions for current sentence
    #     pred  = sequenceLabeler.predict( [(BIO('O'),word,lemma,pos) for label,word,lemma,pos in test_data[n]] )
    #     print pred
        
        
    #print to output file  
    # In Part II, you will have to output predictions on the test set.
    test_data = make_test_data(BIO,testfilename)
    for n in range(N, len(test_data)):
        
        pred  = sequenceLabeler.predict( [(BIO('O'),word,lemma,pos) for offset,word,lemma,pos,mwe,parent,strength,ssense,sid in test_data[n]] )

        with open(outfilename, "a") as myfile:
         for i,label in enumerate(pred):   #pred is a list of labels (i.e. strings) objects (i is index and t is label )
           mwe, ssense = label.split("-")
           test_data[n][i][5] = mwe
           test_data[n][i][8] = ssense
           out = test_data[n][i].join('\t')
           myfile.write(out)
           
         myfile.write('\n')  # adding line break at end of each sentence
           
           
           
           