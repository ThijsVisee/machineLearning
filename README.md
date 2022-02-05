# machineLearning

<https://www.overleaf.com/project/61b086574f545f9b96f1f4b7>

## **How To Run?**

1. Install requirements by running:

    ```
    pip install -r requirements.txt
    ```

2. Run the model:

   ```
   $
   ```

## **Finishing Johann Sebastian Bach's Unfinished Masterpiece**

### **Data**

The data format consists of 4 voices, each encoding the music score as a sequence of integers which represent pitch. The
file data.txt represents each of the 4 voices as a (long, column) vector made from 0's and piano key indices. A typical
part of a voice vector looks like:

```
... 0 0 0 0 56 56 56 56 58 58 58 58 58 58 58 58 70 70 70 70 ....
```

The 0's denote breaks (no sound) and the other integers denote tones. The pitch of a tone is given by the integer, the duration is represented by repeating a
key number. Thus, in the above example, the "58" note is heard twice as long as the "56" note.

### **Pitch Encoding**

In order to explore elementary characteristics of music, we need to encode the data in a richer format, which will allow
the ML algorithm to learn better. The pitch of each note will be encoded in a 5 dimensional vector (Aulon Kuqi thesis).
The pitch vector will have the following format:

```
    [
    0. value proportional to the logarithm of the absolute pitch of the note,
    1. x coordinates of the position of the note in the chroma circle,
    2. y coordinates of the position of the note in the chroma circle,
    3. x coordinates of the note in the circle of fifths,
    4. y coordinates of the note in the circle of fifths,
    ]
```
