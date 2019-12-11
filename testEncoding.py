# coding: utf-8
from sklearn.preprocessing import LabelEncoder
import io, sys
encoder = LabelEncoder()

def printText(array):
    for subArray in array:
        for i in subArray:
            sys.stdout.write(i)
        print("")

X = []
f = io.open("./testInput.txt","r",encoding="utf-8")
fit = []
for line in f:
    t = [char for char in line.lower()[:-1]]
    X.append(t)
    fit.extend(t)

print("Text to encode")
printText(X)
encoder.fit(fit)    
encodedText = []

for tab in X:
    encodedText.append(encoder.transform(tab))
print("Encoded Text")
print(encodedText)

decodedText = []
for tab in encodedText:
    decodedText.append(encoder.inverse_transform(tab))
print("Decoded Text")
printText(decodedText)
