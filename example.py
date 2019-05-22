# define the function blocks
def zero():
    print ("You typed zero.\n")

def sqr():
    print ("n is a perfect square\n")

def even():
    print ("n is an even number\n")

def prime():
    print ("n is a prime number\n")

# map the inputs to the function blocks
options = {'a' : zero,
           1 : sqr,
           4 : sqr,
           9 : sqr,
           2 : even,
           3 : prime,
           5 : prime,
           7 : prime,
}

options['a']()