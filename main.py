# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import ClassificationSystem as cs
import RecommenderSystem as rs
import pandas as pd


def userInput(choice):
    title = input("Inserisci il nome di un titolo che ti è piaciuto ->\n")
    genreC = ""
    characteristicC = ""
    platformC = ""
    year = input("Inserisci l'anno di uscita ->\n")
    publisher = input("Inserisci il publisher ->\n")
    if (choice == '2'):
        genreC = input("Inserisci l'intero corrispondente al genere del gioco :1 Action,2 Action Adventure,3 Action RPG,4 RPG,5 JRPG,6 Rhythm,7 Sports,8 FPS,9 TPS,10 Fighting,11 Racing,12 Platformer,13 Puzzle,14 Strategy,15 Adventure,16 Sandbox,17 MMORPG,18 General,19 Simulation,20 Arcade\n")
        if (genreC == '1'):
            genreC = 'Action'
        elif (genreC == '2'):
            genreC = 'Action Adventure'
        elif (genreC == '3'):
            genreC = 'Action RPG'
        elif (genreC == '4'):
            genreC = 'RPG'
        elif (genreC == '5'):
            genreC = 'JRPG'
        elif (genreC == '6'):
            genreC = 'Rhythm'
        elif (genreC == '7'):
            genreC = 'Sports'
        elif (genreC == '8'):
            genreC = 'FPS'
        elif (genreC == '9'):
            genreC = 'TPS'
        elif (genreC == '10'):
            genreC = 'Fighting'
        elif (genreC == '11'):
            genreC = 'Racing'
        elif (genreC == '12'):
            genreC = 'Platformer'
        elif (genreC == '13'):
            genreC = 'Puzzle'
        elif (genreC == '14'):
            genreC = 'Strategy'
        elif (genreC == '15'):
            genreC = 'Adventure'
        elif (genreC == '16'):
            genreC = 'Sandbox'
        elif (genreC == '17'):
            genreC = 'MMORPG'
        elif (genreC == '18'):
            genreC = 'General'
        elif (genreC == '19'):
            genreC = 'Simulation'
        elif (genreC == '20'):
            genreC = 'Arcade'
    characteristicC = input("Inserisci l'intero corrispondete a una caratteristica distintiva del gioco:1 Open-World,2 Baseball,3 Golf,4 Skateboarding,5 Basketball,6 Soccer,7 Football,8 Snowboarding,9 Tennis,10 Fantasy,11 Sci-Fi,12 Modern,13 Historic,14 Linear,15 Horror,16 3D,17 2D,18 Music,19 GT/Street,20 Rally,21 FormulaOne,22 Motorcycle,23 Car/Combat,24 Combat,25 Kart,26 Flight,27 Stealth,28 Arcade,29 Real-Time,30 Beat-Em-Up,31 Application,32 Ice Hockey,33 VirtualLife,34 General,35 Point-and-Click,36 Snow/Water,37 Tactics,38 Tactical,39 Turn-Based'-> \n")
    if (characteristicC == '1'):
        characteristicC = 'Open-World'
    elif (characteristicC == '2'):
        characteristicC = 'Baseball'
    elif (characteristicC == '3'):
        characteristicC = 'Golf'
    elif (characteristicC == '4'):
        characteristicC = 'Skateboarding'
    elif (characteristicC == '5'):
        characteristicC = 'Basketball'
    elif (characteristicC == '6'):
        characteristicC = 'Soccer'
    elif (characteristicC == '7'):
        characteristicC = 'Football'
    elif (characteristicC == '8'):
        characteristicC = 'Snowboarding'
    elif (characteristicC == '9'):
        characteristicC = 'Tennis'
    elif (characteristicC == '10'):
        characteristicC = 'Fantasy'
    elif (characteristicC == '11'):
        characteristicC = 'Sci-fi'
    elif (characteristicC == '12'):
        characteristicC = 'Modern'
    elif (characteristicC == '13'):
        characteristicC = 'Historic'
    elif (characteristicC == '14'):
        characteristicC = 'Linear'
    elif (characteristicC == '15'):
        characteristicC = 'Horror'
    elif (characteristicC == '16'):
        characteristicC = '3D'
    elif (characteristicC == '17'):
        characteristicC = '2D'
    elif (characteristicC == '18'):
        characteristicC = 'Music'
    elif (characteristicC == '19'):
        characteristicC = 'GT/Street'
    elif (characteristicC == '20'):
        characteristicC= 'Rally'
    elif (characteristicC == '21'):
        characteristicC = 'Formula One'
    elif (characteristicC == '22'):
        characteristicC = '22 Motorcycle'
    elif (characteristicC == '23'):
        characteristicC = 'Car/Combat'
    elif (characteristicC == '24'):
        characteristicC = 'Combat'
    elif (characteristicC == '25'):
        characteristicC = 'Kart'
    elif (characteristicC == '26'):
        characteristicC = 'Flight'
    elif (characteristicC == '27'):
        characteristicC = 'Stealth'
    elif (characteristicC == '28'):
        characteristicC = 'Arcade'
    elif (characteristicC == '29'):
        characteristicC = 'Real-Time'
    elif (characteristicC == '30'):
        characteristicC = 'Beat-Em-Up'
    elif (characteristicC == '31'):
        characteristicC = 'Application'
    elif (characteristicC == '32'):
        characteristicC = 'Ice Hockey'
    elif (characteristicC == '33'):
        characteristicC = 'VirtualLife'
    elif (characteristicC == '34'):
        characteristicC = 'General'
    elif (characteristicC == '35'):
        characteristicC = 'Point-and-Click'
    elif (characteristicC == '36'):
        characteristicC = 'Snow/Water'
    elif (characteristicC == '37'):
        characteristicC = 'Tattics'
    elif (characteristicC == '38'):
        characteristicC = 'Tactical'
    elif (characteristicC == '39'):
        characteristicC = 'Turn-Based'
    platformC = input("Inserisci l'intero corrispondente alla console di gioco:1 Nintendo64,2 PlayStation,3 Dreamcast,4 Wii,5 GameCube,6 PlayStation2,7 Xbox,8 GameBoyAdvance,9 PlayStation3,10 Xbox360,11 PSP,12 DS,13 3DS,14 PlayStationVita,15 WiiU,16 PlayStation4,17 XboxOne,18 Switch,19 PC,20 PlayStation5,21 Xbox Serie X\S-> \n")
    if (platformC == '1'):
        platformC = 'Nintendo64'
    elif (platformC == '2'):
        platformC = 'PlayStation'
    elif (platformC == '3'):
        platformC = 'Dreamcast'
    elif (platformC == '4'):
        platformC = 'Wii'
    elif (platformC == '5'):
        platformC = 'GameCube'
    elif (platformC == '6'):
        platformC = 'PlayStation2'
    elif (platformC == '7'):
       platformC = 'Xbox'
    elif (platformC == '8'):
        platformC = 'GameBoyAdvance'
    elif (platformC == '9'):
        platformC = 'PlayStation3'
    elif (platformC == '10'):
        platformC = 'Xbox360'
    elif (platformC == '11'):
        platformC = 'PSP'
    elif (platformC == '12'):
        platformC = 'DS'
    elif (platformC == '13'):
        platformC = '3DS'
    elif (platformC == '14'):
        platformC = 'PlayStationVita'
    elif (platformC == '15'):
        platformC = 'WiiU'
    elif (platformC == '16'):
        platformC = 'PlayStation4'
    elif (platformC == '17'):
        platformC = 'XboxOne'
    elif (platformC == '18'):
        platformC = 'Switch'
    elif (platformC == '19'):
        platformC = 'PC'
    elif (platformC == '20'):
        platformC = 'PlayStation5'
    elif (platformC == '21'):
        platformC = 'XboxSeriesX\S'
    user_avg = input("Inserisci il voto che daresti a questo gioco ->\n")


    data = {'title':[title],'year_range':[year],'publisher':[publisher],'genre':[genreC],'characteristic':[characteristicC],'platform':[platformC],'user_avg':[user_avg]}
    userInputGame = pd.DataFrame(data)
    return userInputGame


def main():

    choice = menuChoice()
    if choice == '1':
        print("Benvenuto nel sistema di classificazione")
        userInputGame = userInput(choice)
        cs.main(userInputGame)
    elif choice == '2':
        print("Benvenuto nel sistema di raccomandazione")
        userInputGame = userInput(choice)
        rs.main(userInputGame)

def menuChoice():
    print("Benvenuto utente\n")
    choice = input("Premere 1 per scegliere il sistema di classificazione , o 2 per scegliere il sistema di raccomandazione\n")
    return choice

if __name__ == "__main__":
    main()
