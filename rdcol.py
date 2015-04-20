# Takes in Filename, reads file columnwise, and returns dictionary such that:                        
# import rdcol                                                                                       
# a = rdcol.read('filename',headline,datastartline)                                                  
# a["Column_Name"] -> returns the list for that column                                               
#                                                                                                    
# headline and datastartline are always > 0                                                          
#                                                                                                    
# By Dillon Brout                                                                                    
# dbrout@physics.upenn.edu                                                                           

def read(filename,headline,startline,delim=' '):
    linenum = 0
    go = 0
    column_list = []
    return_cols = {}
    inf = open(filename)
    for line in inf:
        #print line
        line = line.replace('#','')
        line = line.strip()
        cols = line.split(delim)
        cols[:] = (value for value in cols if value != '')
        if linenum == headline - 1:
            for col in cols:
                return_cols[col.strip()] = []
                column_list.append(col.strip())
                go = go + 1
            #print column_list
            #print return_cols.keys()
        if (linenum >= startline - 1):
            index = 0
            for col in cols:
                try:
                    return_cols[column_list[index]].append(float(col.strip()))
                except:
                    return_cols[column_list[index]].append(col.strip())
                index = index + 1
        linenum = linenum + 1
    inf.close()
    return return_cols
