import matplotlib.pyplot as plt
from pysnmp.entity.rfc3413.oneliner import cmdgen
import time
import os
from scapy.all import sniff, wrpcap
from scapy.all import sniff, wrpcap, rdpcap

from scapy.all import sniff, wrpcap, rdpcap, IP, TCP, UDP, ICMP

from scapy.arch.windows import get_windows_if_list
import os


import subprocess
from pprint import pprint
from os import path
import re
import paramiko
from scapy.arch.windows import *


capture_dict = dict()
capture_dict["Protocol"] = []
capture_dict["Source IP"] = []
capture_dict["Source Port"] = []
capture_dict["Destination IP"] = []
capture_dict["Destination Port"] = []



class Router():

    def Pre_config():

        global top_talkers_dest
        global top_talkers_source
        global bandwitdh
        
        dosya =open(r"C:\Traffic_Monitor_Capture\Pre_config.txt","w")
                
        average_rate_dst=0
        if len(top_talkers_dest["Destination Port"]) != 0: #for input        
            dosya.write("access-list 110 remark Automation\n")          
            for i in range(0,len(top_talkers_dest["Source IP"])):
                command = ("access-list 110 permit {} host {} host {} eq {}\n".format(top_talkers_dest["Protocol"][i],top_talkers_dest["Source IP"][i],top_talkers_dest["Destination IP"][i],top_talkers_dest["Destination Port"][i]))
                dosya.write(command)
                rate =  top_talkers_dest["Rate"][i]
                average_rate_dst = rate + average_rate_dst
                
            average_rate_dst = int(average_rate_dst / len(top_talkers_dest["Rate"]))
            

            
            print("\n")
            dosya.write("class-map Automation_in\n")
            dosya.write("match access-group 110\n")

            dosya.write("policy-map Automation_in\n")
            dosya.write("class Automation_in\n")

            dosya.write("police rate percent " + str(average_rate_dst) + "\n")
            dosya.write("conform-action transmit\n")
            dosya.write("exceed-action drop\n")
            dosya.write("violate-action drop\n")

        
        top_talkers_dest.clear()
        top_talkers_source.clear()
        dosya.close()

        file = "C:\\Traffic_Monitor_Capture\\Pre_config.txt"
        os.system(file)

    def Config():
        
        #global router_ip
        global control_src
        global control_dst
        global top_talkers_dest
        global top_talkers_source
        
        command =open(r"C:\Traffic_Monitor_Capture\Pre_config.txt","r")

        command_lines = []

        for i in command.readlines():
            a = i.rstrip()
            command_lines.append(a)
        
        router_ip = input("Please enter device ip address:")
        username = input("Please enter username:")
        password = input("Please enter password:")

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(hostname = router_ip,username = username,password = password)
        print("Successful connection", router_ip)

        # ACL config sending
        remote_connection = ssh_client.invoke_shell()
        remote_connection.send("terminal length 0" + "\n")
        remote_connection.send("conf t" + "\n")
        for i in range(0,len(command_lines)):
            remote_connection.send(command_lines[i] + "\n")
        remote_connection.send("end" + "\n")
        #remote_connection.send("write" + "\n")
        remote_connection.send("\n")
        
        remote_connection.send("terminal length 0" + "\n")
        remote_connection.send("sh run" + "\n")
        time.sleep(8)
        output = remote_connection.recv(65535)
        print((output).decode("ascii"))
        ssh_client.close()
        command.close()
        
    
    
    def Top_Talkers():

        global top_talkers_dest
        global top_talkers_source
        global packet_counter_TCP
        global packet_counter_UDP
        global packet_counter_ICMP
        


        print("""
        #======================================= Top Talkers =======================================#
        #                                                                                           #
        #                Select ""TCP_and_UDP_packets.txt"" file for determining Top Talkers        #
        #                                                                                           #
        #===========================================================================================#
        """)
        

        file_directory = os.listdir("C:\\Traffic_Monitor_Capture")
        if file_directory == []:
            print("Folder is empty!")
            Main()
            
        num = 0
        for file in file_directory: #listed files in Traffic_Monitor_Capture folder
            print(str(num)+"." , file)
            num +=1

        while True:
                try:
                    file = int(input("Please select .txt file for router ACL:"))
                    if ".txt" in file_directory[file]:
                        print("You selected %s" % (file_directory[file]))
                        break
                    else:
                        print("\nIt's not .txt file!")
                except ValueError:
                    print("Error!")


        print("Please wait...")
        dosya =open(r"C:\\Traffic_Monitor_Capture\\" +str(file_directory[file]),"r")

        control_src =[]
        control_src.append(capture_dict["Source IP"][0])
        
        control_dst =[]
        control_dst.append(capture_dict["Destination IP"][0])

        
             
        for i in range(0,len(capture_dict["Source IP"])): #Remove reccurance and append to control[]
            for x in range(1,len(capture_dict["Source IP"])):
                if capture_dict.get("Source IP")[x] not in control_src:
                    control_src.append(capture_dict.get("Source IP")[x])
                    break
                
        for i in range(0,len(capture_dict["Destination IP"])): #Remove reccurance and append to control[]
            for x in range(1,len(capture_dict["Destination IP"])):
                if capture_dict.get("Destination IP")[x] not in control_dst:
                    control_dst.append(capture_dict.get("Destination IP")[x])
                    break
                
        average = len(capture_dict["Source IP"])
        
        top_talkers_dest =dict()
        top_talkers_dest["Protocol"] =[]
        top_talkers_dest["Source IP"] = []
        top_talkers_dest["Destination IP"] = []
        top_talkers_dest["Rate"] =[]
        top_talkers_dest["Destination Port"] =[]

        top_talkers_source =dict()
        top_talkers_source["Protocol"] =[]
        top_talkers_source["Source IP"] = []
        top_talkers_source["Destination IP"] = []
        top_talkers_source["Rate"] =[]
        top_talkers_source["Source Port"] =[]
        
        
        print("""
        #======================================= Top Talkers =======================================#
        #                                                                                           #
        #                                                                                           #
        #                                                                                           #
        #===========================================================================================#
        """)


        for a in range(0,len(control_src)): # for each outgoing top talkers
            for b in range(0,len(control_dst)):
                if  control_src[a] != control_dst[b]:
                    counter_UDP = counter_TCP_HTTP = counter_TCP = counter_TCP_HTTPS  =counter_TCP_FTP = counter_UDP_DNS = counter_ICMP = 0
                    counter_UDP_DNS_src = counter_TCP_HTTPS_src = counter_TCP_FTP_src = counter_TCP_HTTP_src = 0
                    for i in range(0,len(capture_dict["Protocol"])):
                        if capture_dict["Source IP"][i] == control_src[a] and capture_dict["Destination IP"][i] == control_dst[b]:# for traffic counters
                            if capture_dict["Protocol"][i] == "17":
                                counter_UDP +=1
                                if capture_dict["Destination Port"][i] == "53":
                                    counter_UDP_DNS +=1
                                if capture_dict["Source Port"][i] == "53":
                                    counter_UDP_DNS_src +=1
                            elif capture_dict["Protocol"][i] == "6":
                                counter_TCP +=1 
                                if capture_dict["Destination Port"][i] == "443":
                                    counter_TCP_HTTPS +=1
                                if capture_dict["Source Port"][i] == "443":
                                    counter_TCP_HTTPS_src +=1
                                if capture_dict["Destination Port"][i] == "21":
                                    counter_TCP_FTP +=1
                                if capture_dict["Source Port"][i] == "21":
                                    counter_TCP_FTP_src +=1
                                if capture_dict["Destination Port"][i] == "80":
                                    counter_TCP_HTTP +=1
                                if capture_dict["Source Port"][i] == "80":
                                    counter_TCP_HTTP_src +=1
                            elif capture_dict["Protocol"][i] == "1":                               
                                counter_ICMP +=1


                    if round((counter_UDP/average)*100,2) >= 20 or round((counter_TCP/average)*100,2) >= 20: ##each source ip generates TCP or UDP
                                print("%s ---> %s"%(control_src[a],control_dst[b]))
                                print("Total UDP packet count for %s is :"%(control_src[a]),counter_UDP," ---> Average over Total Packets:",round((counter_UDP/average)*100,2),"%")
                                print("Total TCP packet count for %s is :"%(control_src[a]),counter_TCP," ---> Average over Total Packets:",round((counter_TCP/average)*100,2),"%")
                                print("HTTP packet count for %s is :" %(control_src[a]), counter_TCP_HTTP)
                                print("HTTPS packet count for %s is :" %(control_src[a]),counter_TCP_HTTPS)
                                print("DNS packet count for %s is :" %(control_src[a]),counter_UDP_DNS )
                                print("FTP packet count for %s is :" %(control_src[a]), counter_TCP_FTP)
                                print("ICMP packet count for %s is :" %(control_src[a]),counter_ICMP)
                                print("\n")
                                

                    try:
                        if round((counter_TCP_HTTP/average)*100,2) >= 15: # for http
                            top_talkers_dest["Protocol"].append("tcp")
                            top_talkers_dest["Destination Port"].append("80")
                            top_talkers_dest["Rate"].append(int((counter_TCP_HTTP/average)*100))
                            top_talkers_dest["Source IP"].append(control_src[a])
                            top_talkers_dest["Destination IP"].append(control_dst[b])
                          
                        if round((counter_TCP_HTTPS/average)*100,2) >= 15: # for https
                            top_talkers_dest["Protocol"].append("tcp")
                            top_talkers_dest["Destination Port"].append("443")
                            top_talkers_dest["Rate"].append(int((counter_TCP_HTTPS/average)*100))
                            top_talkers_dest["Source IP"].append(control_src[a])
                            top_talkers_dest["Destination IP"].append(control_dst[b])
                            
                        if round((counter_UDP_DNS/average)*100,2) >= 15: # for DNS
                            top_talkers_dest["Protocol"].append("udp")
                            top_talkers_dest["Destination Port"].append("53")
                            top_talkers_dest["Rate"].append(int((counter_UDP_DNS/average)*100))
                            top_talkers_dest["Source IP"].append(control_src[a])
                            top_talkers_dest["Destination IP"].append(control_dst[b])

                        if round((counter_TCP_FTP/average)*100,2) >= 20: # for FTP
                            top_talkers_dest["Protocol"].append("tcp")
                            top_talkers_dest["Destination Port"].append("21")
                            top_talkers_dest["Rate"].append(int((counter_TCP_FTP/average)*100))
                            top_talkers_dest["Source IP"].append(control_src[a])
                            top_talkers_dest["Destination IP"].append(control_dst[b])

                        if round((counter_TCP_HTTP_src/average)*100,2) >= 15: # for http src
                            top_talkers_source["Protocol"].append("tcp")
                            top_talkers_source["Source Port"].append("80")
                            top_talkers_source["Rate"].append(int((counter_TCP_HTTP_src/average)*100))
                            top_talkers_source["Source IP"].append(control_src[a])
                            top_talkers_source["Destination IP"].append(control_dst[b])
                          
                        if  round((counter_TCP_HTTPS_src/average)*100,2) >= 15: # for https src
                            top_talkers_source["Protocol"].append("tcp")
                            top_talkers_source["Source Port"].append("443")
                            top_talkers_source["Rate"].append(int((counter_TCP_HTTPS_src/average)*100))
                            top_talkers_source["Source IP"].append(control_src[a])
                            top_talkers_source["Destination IP"].append(control_dst[b])
                            
                        if round((counter_UDP_DNS_src/average)*100,2) >= 15: # for DNS src
                            top_talkers_source["Protocol"].append("udp")
                            top_talkers_source["Source Port"].append("53")
                            top_talkers_source["Rate"].append(int((counter_UDP_DNS_src/average)*100))
                            top_talkers_source["Source IP"].append(control_src[a])
                            top_talkers_source["Destination IP"].append(control_dst[b])

                        if round((counter_TCP_FTP_src/average)*100,2) >= 15: # for FTP src 
                            top_talkers_source["Protocol"].append("tcp")
                            top_talkers_source["Source Port"].append("21")
                            top_talkers_source["Rate"].append(int((counter_TCP_FTP_src/average)*100))
                            top_talkers_source["Source IP"].append(control_src[a])
                            top_talkers_source["Destination IP"].append(control_dst[b])
          
                            
                    except ZeroDivisionError:
                            continue

               
class Traffic():


    def Report():


        file_directory = os.listdir("C:\\Traffic_Monitor_Capture")
    
        if file_directory == []:
            print("Folder is empty!")
            Main()
    
        num =  0
        for file in file_directory: #listed files in Traffic_Monitor_Capture folder
            print(str(num)+".", file)
            num +=1

        while True:
                try:
                    file = int(input("Please select .pcap file for traffic report:"))
                    print("You selected %s" % (file_directory[file]))
                    if ".pcap" in file_directory[file]:
                        break
                    else:
                        print("\nIt's not .pcap file!")
                except ValueError:
                    print("Error!")

    
        print("Traffic report is being prepared..\n")

        capture =rdpcap("C:\\Traffic_Monitor_Capture\\"+ str(file_directory[file]))
        print(capture) # traffic summary
    
        packet_counter = counter_TCP =counter_UDP = counter_others= counter_ICMP= 0
        counter_HTTPS = counter_DNS = counter_HTTP =0
        for packet_counter in range(0,len(capture)): # counting packets for protocols
            packet = capture[packet_counter]
            if ("TCP" in packet):
                counter_TCP +=1
            elif ("UDP" in packet):
                counter_UDP +=1
            elif ("ICMP" in packet):
                counter_ICMP +=1      
            else:
                counter_others +=1

        counter = counter_443_in = counter_443_out = counter_53_in = counter_53_out = counter_80_out = counter_80_in = 0 
        for counter in range(0,len(capture)): # DNS and HTTP
            pkt=capture[counter]
            if pkt.getlayer(IP):
                layer = str(pkt.proto)
                if pkt.getlayer(TCP):
                    if str(pkt.getlayer(IP).sport)== "443":
                        counter_443_in +=1
                    if str(pkt.getlayer(IP).dport) == "443":
                        counter_443_out +=1
                    if str(pkt.getlayer(IP).sport) == "80":
                        counter_80_in +=1
                    if str(pkt.getlayer(IP).dport) == "80":
                        counter_80_out +=1
                elif pkt.getlayer(UDP):
                    if str(pkt.getlayer(IP).sport)== "53":
                        counter_53_in +=1
                    if str(pkt.getlayer(IP).dport) == "53":
                        counter_53_out +=1


        counter_HTTPS = counter_443_out + counter_443_in # total count of HTTPS
        counter_DNS = counter_53_out + counter_53_in # total count of DNS
        counter_HTTP = counter_80_out + counter_80_in # total count of HTTP
        print("HTTPS packet count is :",counter_HTTPS)
        print("HTTP packet count is :",counter_HTTP)
        print("DNS packet count is :",counter_DNS)

        try:
            average_others = (counter_others/len(capture))*100
            average_UDP = (counter_UDP/len(capture))*100
            average_TCP = (counter_TCP/len(capture))*100
            average_ICMP = (counter_ICMP/len(capture))*100
            average_HTTPS = (counter_HTTPS/len(capture))*100
            average_HTTP = (counter_HTTP/len(capture))*100
            average_DNS = (counter_DNS/len(capture))*100

            Total_packet_count = len(capture)
    
            print("Total packet count is: ",len(capture))   

            print("\nTraffic Details:")  
            print("UDP average is: " + "%",round(average_UDP,2))
            print("TCP average is: " + "%",round(average_TCP,2))
            print("ICMP average is: " + "%",round(average_ICMP,2))
            print("HTTPS average is: " + "%",round(average_HTTPS,2))
            print("HTTP average is: " + "%",round(average_HTTP,2))
            print("DNS average is: " + "%",round(average_DNS,2))
    
            print("HTTPS average in TCP " + "%",round((counter_HTTPS / counter_TCP )*100))
            print("HTTP average in TCP " + "%",round((counter_HTTP / counter_TCP )*100))
            print("DNS average in UDP " + "%", round((counter_DNS / counter_UDP )*100))

            labels = "TCP","UDP","ICMP" # protocol based
            sizes = [round(average_TCP,2),round(average_UDP,2),round(average_ICMP,2)]
            explode = (0.1,0.2,0.3)
            figl,axl = plt.subplots()
            axl.pie(sizes,explode=explode, labels=labels,autopct="%0.1f%%",shadow=True,startangle =90)
            axl.axis("equal")

            labels_2 = "HTTPS","DNS","HTTP" # Port based
            sizes_2 = [round(average_HTTPS,2),round(average_DNS,2),round(average_HTTP,2)]
            explode_2 = (0.1,0.2,0.2)
            fig2,axl_2 = plt.subplots()
            axl_2.pie(sizes_2,explode=explode_2, labels=labels_2,autopct="%0.1f%%",shadow=True,startangle =90)
            axl_2.axis("equal")

            #plt.savefig("C:\\Users\\Madness\\Desktop\\traffic.png")
            plt.tight_layout()
            plt.show()
            plt.close()
        except ZeroDivisionError:
            print("Hata!")   
        Main()

    

    def Details():# ACL icin gerekli

        global packet_counter_TCP
        global packet_counter_UDP
        global packet_counter_ICMP
  

        file_directory = os.listdir("C:\\Traffic_Monitor_Capture")
    
        
        if file_directory == []:
            print("Folder is empty!")
            Main()
        
  
        num = 0
        for file in file_directory: #listed files in Traffic_Monitor_Capture folder
            print(str(num)+"." , file)
            num +=1
        while True:
            try:
                file = int(input("Please select .pcap file for traffic report:"))
                print("You selected %s" % (file_directory[file]))
                if ".pcap" in file_directory[file]:
                    break
                else:
                    print("\nIt's not .pcap file!")
            except ValueError:
                print("Error!")
        
        print("Traffic report is being prepared..\n")

        dosya =open(r"C:\Traffic_Monitor_Capture\TCP_and_UDP_packets.txt","w")
    
        capture =rdpcap("C:\\Traffic_Monitor_Capture\\"+str(file_directory[file]))## total packet size
        print("Total Packet count is ",len(capture))

        Total_pcaket_count = len(capture)

        packet_counter_TCP = packet_counter_ICMP =  packet_counter_UDP =0  
        
        for counter in range(0,Total_pcaket_count):
            pkt=capture[counter]
            if pkt.getlayer(IP):
                src_address = pkt.getlayer(IP).src
                dst_address = pkt.getlayer(IP).dst
                layer = str(pkt.proto)
                if pkt.getlayer(TCP):
                    #print("Protocol:" + layer.replace("6","TCP") ,  "Source IP:" + " " + pkt.getlayer(IP).src , str(pkt.getlayer(IP).sport) , "Destination IP:" + " " + pkt.getlayer(IP).dst , str(pkt.getlayer(IP).dport),sep="  ")
                    dosya.write("Protocol:" + layer.replace("6","TCP") + "    " + "Source IP:" + pkt.getlayer(IP).src + "    " + str(pkt.getlayer(IP).sport) + "    " + "Destination IP:" + pkt.getlayer(IP).dst + "    " + str(pkt.getlayer(IP).dport) + "\n")
                    capture_dict["Protocol"].append(layer)
                    capture_dict["Source IP"].append(pkt.getlayer(IP).src)
                    capture_dict["Destination IP"].append(pkt.getlayer(IP).dst)
                    capture_dict["Destination Port"].append(str(pkt.getlayer(IP).dport))
                    capture_dict["Source Port"].append(str(pkt.getlayer(IP).sport))
                    packet_counter_TCP +=1
                
                elif pkt.getlayer(UDP):
                    #print("Protocol:" + layer.replace("17","UDP") + "    " + "Source IP:" + pkt.getlayer(IP).src + "     " + str(pkt.getlayer(IP).sport) + "    " + "Destination IP:" + pkt.getlayer(IP).dst + "    " + str(pkt.getlayer(IP).dport))
                    dosya.write("Protocol:" + layer.replace("17","UDP") + "    " + "Source IP:" + pkt.getlayer(IP).src + "    " + str(pkt.getlayer(IP).sport) + "    " + "Destination IP:" + pkt.getlayer(IP).dst + "    " + str(pkt.getlayer(IP).dport) + "\n")
                    capture_dict["Protocol"].append(layer)
                    capture_dict["Source IP"].append(pkt.getlayer(IP).src)
                    capture_dict["Destination IP"].append(pkt.getlayer(IP).dst)
                    capture_dict["Destination Port"].append(str(pkt.getlayer(IP).dport))
                    capture_dict["Source Port"].append(str(pkt.getlayer(IP).sport))
                    packet_counter_UDP +=1
                    
                elif layer == "1":
                    dosya.write("Protocol:" + layer.replace("1","ICMP") + "    " + "Source IP:" + pkt.getlayer(IP).src + "    " + "Destination IP:" + pkt.getlayer(IP).dst + "\n")
                    capture_dict["Protocol"].append(layer)
                    capture_dict["Source IP"].append(pkt.getlayer(IP).src)
                    capture_dict["Destination IP"].append(pkt.getlayer(IP).dst)
                    capture_dict["Destination Port"].append(" ")
                    capture_dict["Source Port"].append(" ")
                    packet_counter_ICMP +=1
                                                         
        print("Details are saved in C:\Traffic_Monitor_Capture\TCP_and_UDP_packets.txt")



    def Capture():# traffic capture


        interface_list = get_windows_if_list()
        print("\nDevice Interface List:\n")
        number_list=0
        for interface in interface_list:
            print(str(number_list)+".",interface_list[number_list].get("name"))
            number_list +=1

        while True:
                try:
                    select_interface = int(input("Please select an interface for capturing:"))
                    print("You select", interface_list[select_interface].get("name")) # interface selection
                    timeout_val=int(input("\nPlease enter capture duration in seconds:"))
                    break
                except ValueError:
                    print("Error!")
        
        filename = input("Please enter .pcap filename:")
                
        print("Please wait traffic is capturing...")           
        capture = sniff(timeout=timeout_val,iface=interface_list[select_interface].get("name"))#capturing
        wrpcap("C:\\Traffic_Monitor_Capture\\capture_"+filename+".pcap",capture)#saving capture
        print("Traffic was captured as %s" %("capture_"+filename+".pcap"))

        

    def Folder_control():#for save pcap output

        if path.exists(r'C:\Traffic_Monitor_Capture') is False:
            dosya = os.mkdir(r"C:\Traffic_Monitor_Capture")
            print("Working directory created!")


    def Monitor_Interface():

        global bandwitdh

        while True:
                while True:
                    try:
                        answer =input("Default community name is : public , Do you want to change default SNMP value ?")

                        if answer == "yes":
                            community =input("Please enter new community name:")
                            router_ip =input("\nPlease enter router ip address:")
                            break
                        elif answer == "no":
                            community = "public"
                            print("""\nPlease be sure that correct Community name entered on device""")
                            router_ip =input("\nPlease enter router ip address:")
                            break
                    
                    except ValueError:
                        print("Please enter valid input")
                        break
         
                control = subprocess.Popen(["ping ", router_ip],stdout=subprocess.PIPE)
                stdout, stderr = control.communicate()
        
                if control.returncode == 0:
                    print("Device (%s) is reachable" %(router_ip)) #device ping kontrol
                    control.kill()
           
                    interface_list_new =[]
                    cmdGen = cmdgen.CommandGenerator()#SNMP uzerinden interface ifindex getiriyor
                    print ("\nFetching stats for...", router_ip)
                    errorIndication, errorStatus, errorIndex,interface_list= cmdGen.bulkCmd(
                    cmdgen.CommunityData(community),cmdgen.UdpTransportTarget((router_ip, 161)),0,25,'1.3.6.1.2.1.2.2.1.2')

                    for varBindTableRow in interface_list:
                        for name, val in varBindTableRow:
                            print('%s = Interface Name: %s' % (name.prettyPrint(), val.prettyPrint()))
                            interface_list_new.append(val.prettyPrint())

                    interface_name = int(input("\nPlease enter the OID for monitor the interface:"))
                    print("\nYou selected", interface_list_new[interface_name-1])
                
                    errorIndication, errorStatus, errorIndex,interface_bandwidth_value_get = cmdGen.getCmd(
                    cmdgen.CommunityData(community),cmdgen.UdpTransportTarget((router_ip, 161)),'1.3.6.1.2.1.2.2.1.5.'+str(interface_name))#bandwidth

                    for name,val in interface_bandwidth_value_get:
                        bandwitdh = int(val)
                        print("Interface bandwidth is {} bps".format(int(bandwitdh)))
                

                    interval = int(input("\nPlease enter polling interval in second:"))
                    recurrence =int(input("Please enter recurrence count:"))
                    tag = 1
                    response_tag=0

                    while tag <= recurrence :
                        errorIndication, errorStatus, errorIndex,interface_input_value_get = cmdGen.getCmd(cmdgen.CommunityData(community),cmdgen.UdpTransportTarget((router_ip, 161)),'1.3.6.1.4.1.9.2.2.1.1.6.%s' %(interface_name))
                        errorIndication, errorStatus, errorIndex, interface_output_value_get = cmdGen.getCmd(cmdgen.CommunityData(community),cmdgen.UdpTransportTarget((router_ip, 161)),"1.3.6.1.4.1.9.2.2.1.1.8.%s" %(interface_name))
                        for name,val in interface_input_value_get:
                            print("%s Interface Input Traffic = %s bps" % (str(interface_list_new[interface_name-1]),str(val)))
                            interface_input_traffic = int(val)
                        for name,val in interface_output_value_get:
                            print("%s Interface Output Traffic = %s bps" % (str(interface_list_new[interface_name-1]),str(val)))
                            interface_output_traffic = int(val)
                            
                        if interface_input_traffic >= int(bandwitdh*85)/100 :
                            average_in = (interface_input_traffic/bandwitdh)*100
                            print("Interface saturated! It's nearing maximum Rx throughput " +" " + "%",round(average_in,2))
                            response = input("Do you want to capture the traffic ?\n")
                            if response == "yes":
                                Traffic.Capture()
                                user_response = input("Do you want traffic report?")
                                if user_response == "yes":
                                    Traffic.Report()
                                    Main()
                                elif user_response == "no":
                                    tag +=1
                                    if tag > recurrence:
                                        Main()
                                    else:
                                        time.sleep(interval)  
                            elif response == "no":
                                tag +=1
                                if tag > recurrence:
                                    Main()
                                else:
                                    time.sleep(interval)
        
              
                        elif interface_output_traffic >= int(bandwitdh*85)/100 :
                            average_out = (interface_output_traffic/bandwitdh)*100
                            print("Interface saturated! It's nearing maximum Tx throughput " +" " + "%",round(average_out,2))
                            response = input("Do you want to capture the traffic ?\n")
                            if response == "yes":
                                Traffic.Capture()
                                user_response = input("Do you want traffic report?")
                                if user_response == "yes":
                                    Traffic.Report()
                                    Main()
                                elif user_response == "no":
                                    tag +=1
                                    if tag > recurrence:
                                        Main()
                                    else:
                                        time.sleep(interval)       
                            elif response == "no":
                                tag +=1
                                if tag > recurrence:
                                    Main()
                                else:
                                    time.sleep(interval)
                        else:
                            tag +=1
                            if tag > recurrence:
                                Main()
                            else:
                                time.sleep(interval)
                            
                else:
                    print("Router is not reachable!")
                    break
def Main():

    
    while True:

    
        selection=["Network Device Interface Monitor ","Network Traffic Capture","Network Traffic Report","Genarate Traffic Details","Genarate Top Talkers","Write Router Policy-Map","Apply Policy-Map","Exit"]

        Traffic.Folder_control()
    
        select =0
        num=0
        print("""
===================================================================================
=                                                                                 =
=        SSSS  EEEEE  CCCCC  U   U  RRRR   EEEEE  N   N  EEEEE  TTTTT  X   X      =
=       S      E      C      U   U  R   R  E      NN  N  E        T    X X        =
=        SSS   EEEE   C      U   U  RRRR   EEEE   N N N  EEE      T     X         =
=           S  E      C      U   U  R  R   E      N  NN  E        T    X X        =
=       SSSS   EEEEE  CCCCC   UUU   R   R  EEEEE  N   N  EEEEE    T   X   X       =
=                                                                                 =
===================================================================================
=                                                                                 =
===================================================================================
=                                                                                 =
=      SecureNetx Traffic Capture and Analysis Framework                          =
=                                                                                 =
=        - Real-time Traffic Capture and Analysis                                 =
=        - Packet Header Details Extraction                                       =
=        - Threshold Trigger Detection and Response                               =
=        - Top Talkers Identification                                             =
=        - ACL Management and Configuration                                       =
=        - Integration with Network Devices                                       =
=                                                                                 =
===================================================================================

    
               """)
        for num in selection:
            print(str(select)+".",selection[select])
            select +=1
    
        try:

            select = int(input("Select an option to continue Monitoring and Analysis "))
        
            if select == 7:
                exit()
            elif select == 0:
                print("\nYou selected",selection[select])
                Traffic.Monitor_Interface()
            elif select == 1:
                print("\nYou selected",selection[select])
                Traffic.Capture()
            elif select == 2:
                print("\nYou selected",selection[select])
                Traffic.Report()
            elif select == 3:
                print("\nYou selected",selection[select])
                Traffic.Details()
            elif select == 4:
                print("\nYou selected",selection[select])
                Router.Top_Talkers()
            elif select == 5:
                print("\nYou selected",selection[select])
                Router.Pre_config()
            elif select == 6:
                print("\nYou selected",selection[select])
                Router.Config()
            else:
                print("\nYou have entered wrong number!")
        except ValueError:
            print("Please enter valid number!")

Main()
 