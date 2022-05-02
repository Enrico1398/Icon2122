# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import RecommenderSystem as rs
import string

def userInput():
    userInputGame = input("Inserisci il nome di un titolo che ti è piaciuto\n")
    return userInputGame


def main():

    choice = menuChoice()
    if choice == '1':
        print("Benvenuto nel sistema di classificazione")
    elif choice == '2':
        print("Benvenuto nel sistema di raccomandazione")
        userInputGame = userInput()
        rs.main()

def menuChoice():
    print("Benvenuto utente\n")
    choice = input("Premere 1 per scegliere il sistema di classificazione , o 2 per scegliere il sistema di raccomandazione\n")
    return choice

if __name__ == "__main__":
    main()
