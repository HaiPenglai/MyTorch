import packageA

packageA.greet()
packageA.hello()
packageA.packageB.bye()
# packageA.bye()

print(packageA.__name__) # packageA
print(packageA.__file__) # example\my_package\packageA\__init__.py

'''
from packageA import *

hello()
greet()
bye()
'''