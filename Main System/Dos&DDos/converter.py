import dpkt
import csv
import time

def pcap_to_csv(input_pcap_file, output_csv_file):
    with open(input_pcap_file, 'rb') as pcap_file, open(output_csv_file, 'w', newline='') as csv_file:
        pcap = dpkt.pcap.Reader(pcap_file)
        csv_writer = csv.writer(csv_file)

        # Write the header row
        header = ['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
                  'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max', 'Fwd Pkt Len Min',
                  'Fwd Pkt Len Mean', 'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
                  'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
                  'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot', 'Fwd IAT Mean',
                  'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Tot', 'Bwd IAT Mean',
                  'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
                  'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
                  'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
                  'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
                  'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio',
                  'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
                  'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg', 'Bwd Pkts/b Avg',
                  'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts', 'Subflow Bwd Pkts',
                  'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
                  'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
                  'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
        csv_writer.writerow(header)

        flow_dict = {}
        for ts, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    flow_key = (dpkt.utils.inet_to_str(ip.dst), dpkt.utils.inet_to_str(ip.src), tcp.dport, tcp.sport, ip.p)
                    if flow_key in flow_dict:
                        flow = flow_dict[flow_key]
                        flow['Tot Fwd Pkts'] += 1
                        flow['Tot Bwd Pkts'] += 1
                        flow['TotLen Fwd Pkts'] += len(tcp.data)
                        flow['TotLen Bwd Pkts'] += 0  # Change this for BWD data
                    else:
                        flow_dict[flow_key] = {
                            'Dst Port': tcp.dport,
                            'Protocol': ip.p,
                            'Timestamp': time.strftime('%d/%m/%Y %H:%M:%S', time.gmtime(ts)),
                            'Flow Duration': 0,
                            'Tot Fwd Pkts': 1,
                            'Tot Bwd Pkts': 1,
                            'TotLen Fwd Pkts': len(tcp.data),
                            'TotLen Bwd Pkts': 0,  # Change this for BWD data
                            'Fwd Pkt Len Max': len(tcp.data),
                            'Fwd Pkt Len Min': len(tcp.data),
                            'Fwd Pkt Len Mean': len(tcp.data),
                            'Fwd Pkt Len Std': 0,
                            'Bwd Pkt Len Max': 0,  # Change this for BWD data
                            'Bwd Pkt Len Min': 0,  # Change this for BWD data
                            'Bwd Pkt Len Mean': 0,  # Change this for BWD data
                            'Bwd Pkt Len Std': 0,  # Change this for BWD data
                            # Add other fields and calculations here
                        }

        # Write flow data to CSV
        for flow in flow_dict.values():
            csv_writer.writerow([flow.get(field, 0) for field in header])

if __name__ == '__main__':
    input_pcap_file = 'capture_www.pcap'  # Replace with the path to your PCAP file
    output_csv_file = 'output.csv'  # Replace with the desired CSV file path
    pcap_to_csv(input_pcap_file, output_csv_file)
