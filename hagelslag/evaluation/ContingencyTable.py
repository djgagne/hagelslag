import numpy as np

class ContingencyTable:
    def __init__(self,a,b,c,d):
        """Initializes a contingency table of the following form:
                  Event
                Yes No
Forecast    Yes  a  b   
            No   c  d
        """
        self.table = np.array([[a,b],[c,d]],dtype=float)
        self.N = self.table.sum()
    def update(self,a,b,c,d):
        self.table.ravel()[:] = [a,b,c,d]
        self.N = self.table.sum()
        
    def POD(self):
        """Returns Probability of Detection (POD) or Hit Rate.
           Formula:  a/(a+c)"""
        return self.table[0,0]/(self.table[0,0] + self.table[1,0])
    
    def FOH(self):
        """Returns Frequency of Hits (FOH) or Success Ratio.
           Formula:  a/(a+b)"""
        return self.table[0,0]/(self.table[0,0] + self.table[0,1])

    def FAR(self):
        """Returns False Alarm Ratio (FAR).
           Formula:  b/(a+b)"""
        return self.table[0,1]/(self.table[0,0] + self.table[0,1])
    def POFD(self):
        """Returns Probability of False Detection (POFD).
           b/(b+d)"""
        return self.table[0,1]/(self.table[0,1] + self.table[1,1])

    def FOM(self):
        """Returns Frequency of Misses (FOM).
           Formula:  c/(a+c)."""
        return self.table[1,0]/(self.table[0,0] + self.table[1,0])

    def DFR(self):
        """Returns Detection Failure Ratio (DFR).
           Formula:  c/(c+d)"""
        return self.table[1,0]/(self.table[1,0] + self.table[1,1])

    def PON(self):
        """Returns Probability of Null (PON).
           Formula:  d/(b+d)"""
        return self.table[1,1]/(self.table[0,1] + self.table[1,1])

    def FOCN(self):
        """Returns Frequency of Correct Null (FOCN).
           Formula:  d/(c+d)"""
        return self.table[1,1]/(self.table[1,0] + self.table[1,1])

    def Bias(self):
        """Returns Bias.  Formula:  (a+b)/(a+c)"""
        return (self.table[0,0] + self.table[0,1])/(self.table[0,0] + self.table[1,0])

    def Accuracy(self):
        """Finley's measure, fraction correct, accuracy (a+d)/N"""
        return (self.table[0,0] + self.table[1,1])/self.N

    def CSI(self):
        """Gilbert's Score or Threat Score or Critical Success Index a/(a+b+c)"""
        return self.table[0,0]/(self.table[0,0] + self.table[0,1] + self.table[1,0])

    def ETS(self):
        """Equitable Threat Score, Gilbert Skill Score, v, (a - R)/(a + b + c - R), R=(a+b)(a+c)/N"""
        R = (self.table[0,0] + self.table[0,1]) * (self.table[0,0] + self.table[1,0])/self.N
        return (self.table[0,0] - R)/(self.table[0,0] + self.table[0,1] + self.table[1,0] - R)

    def HSS(self):
        """Doolittle (Heidke) Skill Score.  2(ad-bc)/((a+b)(b+d) + (a+c)(c+d))""" 
        return 2*(self.table[0,0] * self.table[1,1] - self.table[0,1] * self.table[1,0])/((self.table[0,0] + self.table[0,1]) * (self.table[0,1] + self.table[1,1]) + (self.table[0,0] + self.table[1,0]) * (self.table[1,0] + self.table[1,1]))

    def PSS(self):
        """Peirce (Hansen-Kuipers, True) Skill Score (ad - bc)/((a+c)(b+d))"""
        return (self.table[0,0] * self.table[1,1] - self.table[0,1] * self.table[1,0])/((self.table[0,0] + self.table[1,0]) * (self.table[0,1] + self.table[1,1]))
    def CSS(self):
        """Clayton Skill Score (ad - bc)/((a+b)(c+d))"""
        return (self.table[0,0] * self.table[1,1] - self.table[0,1] * self.table[1,0])/((self.table[0,0] + self.table[0,1]) * (self.table[1,0] + self.table[1,1]))
    def __str__(self):
        table_string = '\tEvent\n\tYes\tNo\nYes\t%d\t%d\nNo\t%d\t%d\n' % (self.table[0,0],self.table[0,1],self.table[1,0],self.table[1,1])
        return table_string

if __name__ == "__main__":
    table = ContingencyTable(1,2,3,4)
    print table
    print getattr(table,"PSS")()
    
