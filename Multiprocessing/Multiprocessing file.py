#!/usr/bin/env python
# coding: utf-8

# In[3]:


from multiprocessing import Process, Queue, Lock
import NN
import random


# In[18]:


if __name__ == "__main__":
    result = Queue()
    P1 = Process(target=NN.NN, args=(random.randint(1, 6000), result))
    P2 = Process(target=NN.NN, args=(random.randint(1, 6000), result))
    P3 = Process(target=NN.NN, args=(random.randint(1, 6000), result))
    P4 = Process(target=NN.NN, args=(random.randint(1, 6000), result))
    P1.start()
    P2.start()
    P3.start()
    P4.start()
    P1.join()
    P2.join()
    P3.join()
    P4.join()
    P1.terminate()
    P2.terminate()
    P3.terminate()
    P4.terminate()
    output0 = result.get()
    output1 = result.get()
    output2 = result.get()
    output3 = result.get()
    print(output)
    print(output2)
    print(output3)
    print(output4)


# In[ ]:




