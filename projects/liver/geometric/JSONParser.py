import json


class JSONParser:

    def __init__(self, filename):

        with open(filename, "r") as f:
            self.json_file = json.load(f)

    def getLiversName(self):
        return self.json_file.values()

    def getAllLivers(self):
        return self.json_file

    def getAllLiversWithPV(self):
        temp = {k: v for k,v in zip(self.json_file, self.json_file.values()) if v}
        return temp

    def getAllLineCoefficents(self):
        temp = [r["line"] for liver in self.json_file.values() if liver for r in liver["lines"] ]
        return temp

    def getAllPVSlice(self):
        temp = [{"left_pv": liver["left_pv"], "right_pv": liver["right_pv"]} for liver in self.json_file.values() if liver]
        return temp

    def getLiverNameFromIdx(self, idx):
        return list(self.json_file.keys())[idx]

    def getLen(self):
        return len(self.json_file)


# parser = JSONParser("D:\\Universita\\Laurea Magistrale - Computer Science Engeneering\\Tesi\\LiverSegmentation\\results2.json")
# g = parser.getAllLiversWithPV()
# h = parser.getAllLineCoefficents()
# n = parser.getAllPVSlice()
# print(parser.getLiverNameFromIdx(1))