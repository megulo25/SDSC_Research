# Import modules
from bs4 import BeautifulSoup
from splinter import Browser
import pymongo
import time
import os
#----------------------------------------------------------------------#
# Establish Connection to MongoDB
# Connect to MongoDB
conn = 'mongodb://localhost:27017'

# Create client
client = pymongo.MongoClient(conn)

# Connect to db
db = client.Birds
print('Connected to database!')
#----------------------------------------------------------------------#
# Utility Functions:
'Below are functions used to facilitate the scraping process'

def startBrowser(url):
    """
    Initializes the chrome headless browser.

    Arguments:
    url -- string, link to landing page.

    Returns:
    browser -- Splinter object, an object containing information about the chrome browser.
    """

    # Inialiaze headless browser
    executable_path = {'executable_path': os.path.join("driver","chromedriver")}
    browser = Browser('chrome', **executable_path, headless=False)
    print("Initialized the headless browser!\n")
    
    # URl
    browser.visit(url)
    print("Visited the site!\n")
    return browser
    
def clean_class_list(list_):
    """
    Filtering class list to get search queries from 'classes.txt' file.

    Argument:
    list_ -- list, Contains a list of each row in the 'classes.txt' file.

    Returns:
    new_list = list, Contains only the class names to be used as search queries.
    """
    new_list = []

    # Loop and clean
    for list_item in list_:
        # Split
        split_item = list_item.split()
        # Keep only the birds name
        joined_list = ' '.join(split_item[1:])
        new_list.append(joined_list)
    return new_list

def getData(browser, soup):
    """
    This function grabs all the necessary information from the information page of 
    a specific bird. Once landed on the page containing information about a specific
    bird the following are collected:
        - Name
        - Order
        - Family
        - Habitat
        - Food
        - Nesting
        - Behavior
        - Conservation

    Arguments:
    browser -- Splinter object, an object containing information about the chrome browser.
    soup -- Beautiful Soup object, Used to parse html on the browser.

    Returns:
    dict_ -- Dictionary, Contains the aforementioned information about a bird. Each key 
    corresponds to (name, order, family, etc) and the value are collected from the webpage.
    """
    dict_ = {}
    ul = soup.find_all('ul', class_='additional-info')
    lis = ul[0].find_all('li')
    
    # Name of bird
    name_div = soup.find('div', class_='species-info')
    name = name_div.find('h4').get_text()
    dict_['NAME'] = name
    
    # Get the Order
    li_order = lis[0]
    li_order_txt = li_order.get_text()
    li_order_split = li_order_txt.split()
    dict_['ORDER'] = li_order_split[1]
    
    # Get the Family
    li_family = lis[1]
    li_family_txt = li_family.get_text()
    li_family_split = li_family_txt.split()
    dict_['FAMILY'] = li_family_split[1]
    
    
    ul_lh_menu = soup.find_all('ul', class_='LH-menu')
    lis_lh_menu = ul_lh_menu[0].find_all('li')
    
    # Get the Habitat
    habitat_tag = lis_lh_menu[0].find_all('a', class_='text-label')
    habitat_text = habitat_tag[0].get_text()
    habitat = habitat_text[7:]
    dict_['HABITAT'] = habitat
    
    # Get the Food
    food_tag = lis_lh_menu[1].find_all('a', class_='text-label')
    food_text = food_tag[0].get_text()
    food = food_text[4:]
    dict_['FOOD'] = food
    
    # Get the Nesting
    nesting_tag = lis_lh_menu[2].find_all('a', class_='text-label')
    nesting_text = nesting_tag[0].get_text()
    nesting = nesting_text[7:]
    dict_['NESTING'] = nesting
    
    # Get the Behavior
    behavior_tag = lis_lh_menu[3].find_all('a', class_='text-label')
    behavior_text = behavior_tag[0].get_text()
    behavior = behavior_text[8:]
    dict_['BEHAVIOR'] = behavior
    
    # Get the Conservation
    conservation_tag = lis_lh_menu[4].find_all('a', class_='text-label')
    conservation_text = conservation_tag[0].get_text()
    conservation = conservation_text[12:]
    dict_['CONSERVATION'] = conservation
    
    return dict_
#----------------------------------------------------------------------#
# Main script

if __name__ == "__main__":
    
    # Get a list of all the birds
    with open('classes.txt', 'r') as file_reader:
        class_list = file_reader.readlines()

    # Clean the list
    cleaned_list = clean_class_list(class_list)
    cleaned_list = cleaned_list[1:]

    # Temp
    cleaned_list = cleaned_list[372:]

    # Open up the chrome browser
    url = 'https://www.allaboutbirds.org/search/'
    print('Starting chrome browser')
    browser = startBrowser(url)

    # Loop through each item in the list
    num_searches = len(cleaned_list)
    current_search_index = 0
    for item in cleaned_list:
        print('#####################################################################################')
        print('Search: {0}/{1}'.format(current_search_index, num_searches))
        print('Getting information on: {0}\n'.format(item))
        current_search_index +=1

        # Input Search query
        browser.find_by_tag('form').first.find_by_tag('input').fill(item)

        # Submit query
        browser.find_by_tag('form').first.find_by_tag('button').first.click()

        are_there_more = True
        num_problems = 0
        while are_there_more:
            time.sleep(2)
            try:
                # Click 'See More Birds' to get an exhaustive list of birds
                browser.find_by_id('btn-guide-more').click()
            except:
                print('Attempt {0}/5 to click for more birds'.format(num_problems))
                num_problems+=1

                if num_problems > 5:
                    print('\nThis is all of the birds visible.')
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
        print('Looping through each kind of: {0}'.format(item))
        for href in href_list:
            browser.visit(href)

            # Get the current html
            info_page_html = browser.html
            info_soup = BeautifulSoup(info_page_html, 'html.parser')

            # Get the data
            try:
                dict_ = getData(browser, info_soup)
            except:
                print('Error with {0}, moving on...'.format(href))

            # Add to Mongodb
            db.bird_info.insert_one(dict_)
        print('Loop completed! Moving on!')
        # Go back to search page
        browser.visit(url)
