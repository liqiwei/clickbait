import requests
import json
import csv

def get_headlines(source):
    url = "https://api.cognitive.microsoft.com/bing/v7.0/news/search"
    querystring = {"q":"site:"+str(source),"count":"100"}
    headers = {
    'content-type': "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW",
    'ocp-apim-subscription-key': "500ebfb9c07f48fab40d88aeee2626e9",
    'cache-control': "no-cache"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)
    my_json = json.loads(response.text)

    #set up file
    file_name = source
    header1 = "source"
    header2 = "title"
    #all files written to .idea!
    with open(r"/Users/qiweili/Desktop/all_sources_headlines/"+file_name + ".csv", 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow([header1,header2])

        for key,value in my_json.items():
            if key =="value":
                for key in value:
                    headline = key.get('name')
                    print(headline)
                    writer.writerow([file_name, headline])
    file.close()

def main():
    num=1
    with open(r"/Users/qiweili/Desktop/other_clickbait/all_unique_sources.csv","r") as readfile:
        reader=csv.reader(readfile, delimiter =',')
        for row in reader:
            source = row[0]
            if source.startswith('www.'):
                source = source[4:]
            print(num)
            num=num+1
            print("the source is:"+source)
            get_headlines(source)

main()