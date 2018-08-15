from bs4 import BeautifulSoup
from splinter import Browser
import os

def startBrowser(url):
    # Inialiaze headless browser
    executable_path = {'executable_path': os.path.join("driver","chromedriver")}
    browser = Browser('chrome', **executable_path, headless=False)
    print("Initialized the headless browser!\n")
    
    # URl
    browser.visit(url)
    print("Visited the site!\n")
    return browser

def clean_class_list(list_):
    new_list = []
    # Loop and clean
    for list_item in list_:
        # Split
        split_item = list_item.split()
        # Keep only the birds name
        joined_list = ' '.join(split_item[1:])
        new_list.append(joined_list)
    return new_list

def getData(browser):
    'Return as Dictionary'
    # Get the Order
    
    # Get the Family
    
    # Get the Habitat
    
    # Get the Food
    
    # Get the Nesting
    
    # Get the Behavior
    
    # Get the Conservation
    
    # Grab the cool facts (optional)
    
    # Consider getting more info from 'ID info'
    pass

if __name__ == '__main__':

    # Get a list of all the birds
    with open('classes.txt', 'r') as file_reader:
        class_list = file_reader.readlines()

    # Clean the list
    cleaned_list = clean_class_list(class_list)
    cleaned_list = cleaned_list[1:]

    # Open up the chrome browser
    url = 'https://www.allaboutbirds.org/search/'
    print('Starting chrome browser')
    browser = startBrowser(url)


    # Loop through each item in the list
    for item in cleaned_list:

    # Input Search query
    browser.find_by_tag('form').first.find_by_tag('input').fill(item)

    # Submit query
    browser.find_by_tag('form').first.find_by_tag('button').first.click()
    
    are_there_more = True
    num_problems = 0
    while are_there_more:
        
        try:
            # Click 'See More Birds' to get an exhaustive list of birds
            browser.find_by_id('btn-guide-more').click()
        except:
            print('Theres a problem')
            num_problems+=1
            
            if num_problems > 5:
                break

    # Get list of birds items using soup
    'Soup will get each link from the <a> tag inside each <li>'
    current_html = browser.html
    soup = BeautifulSoup(current_html, 'html.parser')
    
    # Get all the <a> tags
    a_list = soup.find_all('a', class_='audio-img', href=True)
    
    # Grab all the href from the <a> tags
    href_list = []
    for a in a_list:
        href_list.append(a.get('href'))
        
    # Explore those sites and gather the necessary info.
    for href in href_list:
        browser.visit(href)
        
        # Get the data
        dict_ = getData(browser)
        
        # Add to Mongodb
        
        
        break
    break