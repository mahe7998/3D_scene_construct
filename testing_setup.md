Command to download more objects in database:

Step to test new Object database:

docker-compose --% run --rm --remove-orphans tests python3 -c "from src.utils.database import Database; db = Database('/data/database/assets.db'); objs = db.get_all_objects(); print('Objects in DB:', len(objs)); [print(' ',  o['id'], ':', o['file_path']) for o in objs[:3]]"