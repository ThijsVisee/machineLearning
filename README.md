# machineLearning
https://www.overleaf.com/project/61b086574f545f9b96f1f4b7

**Finishing	Johann Sebastian	Bach's	Unfinished	Masterpiece**

Data:

The data format consits of 4 voices, each encoding the music score as a sequence of integers which represent pitch. The file data.txt represents each of the 4
voices as a (long, column) vector made from 0's and piano key indices. A typical part of a voice vector looks like:
```
... 0 0 0 0 56 56 56 56 58 58 58 58 58 58 58 58 70 70 70 70 ....
```
The 0's denote breaks (no sound) and the other integers denote tones. The pitch of a tone is given by the integer, the duration is represented by repeating a
key number. Thus, in the above example, the "58" note is heard twice as long as the "56" note.
