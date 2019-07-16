import sqlite3
import read_model as rm
import numpy as np

def create_database(database_name='data/images_points.db'):
    points3d = rm.read_points3D_text('data/points3D.txt')
    print('%d 3d points'%len(points3d))
    #cameras = rm.read_cameras_text('data/cameras.txt')
    #print('%d cameras'%len(cameras))
    images = rm.read_images_text('data/images.txt')
    print('%d images'%len(images))

    c = sqlite3.connect(database_name)
    try:
        res = c.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='global_features';")
        if(res.fetchone()[0] == 1):
            print('Database already exists - Please either delete or use other name')
            exit()
        else:
            c.execute('''CREATE TABLE images (image_id INTEGER PRIMARY_KEY NOT NULL, qvec BLOB, tvec BLOB, camera_id INTEGER, name TEXT, data_cols INTEGER, xys BLOB, point3D_ids BLOB)''')
            #Point3D(id=1, xyz=array([ 650.802  ,   -2.78686, -287.018  ]), rgb=array([87, 87, 87]), error=1.0, image_ids=array([226, 222, 220, 218]), point2D_idxs=array([13334, 14063, 11416, 10382]))
            c.execute('''CREATE TABLE points3d (point_id INTEGER PRIMARY_KEY NOT NULL, xyz BLOB, image_id_cols INTEGER, image_ids BLOB)''')
    except sqlite3.Error as e:
        print(e)
        exit()

    print('Insert images')
    for image_id in images.keys():
        img = images[image_id]
        c.execute("INSERT INTO images VALUES (?,?,?,?,?,?,?,?)", [image_id, img.qvec, img.tvec, img.camera_id, img.name, img.point3D_ids.shape[0], img.xys, img.point3D_ids])

    print('Insert points')
    for p_id in points3d.keys():
        pt = points3d[p_id]
        c.execute("INSERT INTO points3d VALUES (?,?,?,?)", [p_id, pt.xyz, pt.image_ids.shape[0], pt.image_ids])
    
    c.commit()    
    c.close()
    print('Finished')
    
def read_database(path='data/images_points.db'):
    images = {}
    points = {}
    c = sqlite3.connect(path).cursor()
    for row in c.execute('SELECT image_id, qvec, tvec, camera_id, name, data_cols, xys, point3D_ids FROM images;'):
        image_id = row[0]
        qvec = np.frombuffer(row[1], dtype=np.float64).reshape(4)
        tvec = np.frombuffer(row[2], dtype=np.float64).reshape(3)
        xys = np.frombuffer(row[6], dtype=np.float64).reshape(-1, 2)
        point3D_ids = np.frombuffer(row[7], dtype=np.int64).reshape(-1)
        images[image_id] = rm.Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=row[3], name=row[4],
                    xys=xys, point3D_ids=point3D_ids)
    for row in c.execute('SELECT point_id, xyz, image_id_cols, image_ids FROM points3d;'):
        point3D_id = row[0]
        xyz = np.frombuffer(row[1], dtype=np.float64).reshape(3)
        image_ids = np.frombuffer(row[3], dtype=np.int64).reshape(-1)
        points[point3D_id] = rm.Point3D(id=point3D_id, xyz=xyz, rgb=None,
                                               error=None, image_ids=image_ids,
                                               point2D_idxs=None)
    return images, points
    
    
if __name__ == '__main__':
    create_database()