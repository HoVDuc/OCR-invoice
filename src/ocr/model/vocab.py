class Vocab():
    def __init__(self, chars):
        self.pad = 0
        self.sos = 1
        self.eos = 2
        self.mask_token = 3

        self.chars = chars

        self.c2i = {c:i+4 for i, c in enumerate(chars)}
        self.i2c = {i+4:c for i, c in enumerate(chars)}
                
        self.i2c[0] = '<pad>'
        self.i2c[1] = '<sos>'
        self.i2c[2] = '<eos>'
        self.i2c[3] = '*'

    def encode(self, chars):
        encod = []
        for c in chars:
            try: 
                encod.append(self.c2i[c])
            except KeyError:
                encod.append(self.mask_token)
        try:
            return [self.sos] + encod + [self.eos]
        except:
            print('*')
        
    def decode(self, ids):
        first = 1 if self.sos in ids else 0
        last = ids.index(self.eos) if self.eos in ids else None
        sent = ''.join([self.i2c[i] for i in ids[first:last]])
        return sent
    
    def __len__(self):
        return len(self.c2i) + 4
    
    def batch_decode(self, arr):
        texts = [self.decode(ids) for ids in arr]
        return texts

    def __str__(self):
        return self.chars
