from website import create_app

app = create_app()

##to run the server
if __name__ == '__main__': #only if you run this file directly  (not if import), we execute this line
    app.run(debug = True)

 