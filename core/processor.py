#!/usr/bin/env python
# -*- coding: utf-8 -*-
from indexing import *
from debugger.logging import Logger
from debugger.util.math import *
from debugger.conf import *
from vispy.gloo import IndexBuffer
import numpy as np
import itertools
from threading import Lock

log = Logger(PROCESSOR_LOG_FILE, V_DEBUG, real_time = True)
reshape_table = {0: (28,28),
                 1: (4, 5)
                }

### Need a State class?

class Panes(object):
    def __init__(self):
        self.data = {}

    def add_target(self, target, old_item, new_item):
        """
            DOC
        """
        tex_ids     = range(old_item, new_item)
        positions   = range(old_item * 4, new_item * 4)
        tex_indices = range(old_item * 4, new_item * 4)
        indices     = range(old_item * 6, new_item * 6)
        all_indices = tex_ids, positions, indices, tex_indices
        self.data[target] = all_indices
        log.info('[MAIN] Pane added for target {}'.format(target))

    def get_texture_id(self, target):
        """
            DOC
        """
        if target not in self.data:
            return None
        else:
            return self.data[target][0]

    def registered_targets(self):
        """
            DOC
        """
        return self.data.keys()

    def check_integrity(self):
        """
            DOC
        """
        data = [self.get_texture_id(layer) for layer in self.registered_targets()]
        return is_contiguous_set(data)

class Processor(object):
    """
        The processor handles all the Preprocessing of the data before display.
        the controler sends it the model structure and the orders to display.
    """
    def __init__(self, targets, struct):
        self.has_updated     = False
        self.targets        = targets
        self.struct         = struct
        self.positions      = None
        self.indices        = None
        self.textures       = None
        self.texindex       = None
        self.panes          = Panes()
        self.n_item         = 0
        self.lock           = Lock()
        self.state          = 'numb'
        log.info('[MAIN] Processor Initialised')

    def update_panes(self, target):
        log.info('[MAIN] Updating panes')
        layer_id    = target[0]
        n_item      = self.struct[layer_id][0]
        shape       = self.struct[layer_id][1]
        vertices, indices, texindex = create_table(n_item)
        log.info('[MAIN] tables created')
        none_set = not any((self.positions is not None, self.indices is not None,self.textures is not None, self.texindex is not None))
        all_set  = all((self.positions is not None, self.indices is not None,self.textures is not None, self.texindex is not None))
        if none_set:
            log.info("First Display")
            n_old_item         = self.n_item
            new_positions      = vertices
            new_indices        = indices
            new_texindex       = texindex
            textures           = np.random.rand(n_item, *shape).astype(np.float32) - 0.5
            # For now we consider all textures should be resized to fit the input layer format
            # This should be updated when new sampling methods are done
            new_texture        = resize_nearest(textures, reshape_table[0])
        elif not all_set:
            log.error('processor positions, indices textures and texindex are set for some but not for all')
        else:
            log.info('Adding new display frames')
            # TODO: Reallocation scheme.
            # For now we only split the screen vertically into two equal regions
            # But Have a 3D visualization of the model could be cool
            old_vertices    = reallocate_table(self.positions, np.asarray((2, 1)), np.asarray((0, 0)))
            new_vertices    = reallocate_table(vertices, np.asarray((2, 1)), np.asarray((1, 0)))
            new_positions   = np.concatenate((old_vertices, new_vertices), axis = 0)

            # Fuse drawing and texture indices
            n_old_item      = self.n_item
            indices         = shift_indices(indices, n_old_item * 4)
            new_indices     = np.concatenate((self.indices, indices), axis = 0)
            new_texindex    = np.concatenate((self.texindex, texindex), axis = 0)
            adjust_texindex(new_texindex)

            # Init and fuse textures
            dimz = [n_item] + list(shape)
            textures = np.zeros(dimz, dtype = np.float32) - 0.5
            new_texture = resize_nearest(textures, reshape_table[0])
            new_texture = np.concatenate((self.textures, new_texture), axis = 0)

        # Apply Computed changes
        # TODO: Should lock here
        self.positions      = new_positions.astype(np.float32)
        self.indices        = new_indices.astype(np.uint32)
        self.index_buffer   = IndexBuffer(self.indices)
        self.texindex       = new_texindex.astype(np.float32)
        self.textures       = new_texture.astype(np.float32)
        self.n_item         = self.n_item + n_item
        self.panes.add_target(target, n_old_item, self.n_item)

        # Notify it is ready for change pane display update
        self._log_shapes()
        self.has_updated = True

    def _log_shapes(self):
        """
            Verbose logging of the indices and textures shape + integrity
        """
        log.verb("self.positions.shape= {}".format(self.positions.shape))
        log.verb("self.indices.shape= {}".format(self.indices.shape))
        log.verb("self.texindex.shape= {}".format(self.texindex.shape))
        log.verb("self.textures.shape= {}".format(self.textures.shape))
        log.verb("integrity= {}".format(self._check_integrity()))

    def _check_integrity(self):
        """
            Returns True if all indices, textures and panes have coherent shapes.
            Returns False otherwise
        """
        try:
            c = [False, False, False, False, False]
            c[0] = self.positions.shape   == (self.n_item * 4, 2)
            c[1] = self.indices.shape     == (self.n_item * 6,)
            c[2] = self.texindex.shape    == (self.n_item * 4, 3)
            c[3] = self.textures.shape    == (self.n_item, reshape_table[0][0], reshape_table[0][1])
            c[4] = self.panes.check_integrity()
        except AttributeError:
            log.error('Check integrity failed with on AttributeError'.format(c))
        if not all(c):
            log.error('Check integrity failed with status {}. Setting state to numb'.format(c))
            self.state = 'numb'
        return all(c)

    def order(self, target, layers):
        """
            This method should set the number of display frames.
            Returns N, the number of frames to be displayed and transmits it
            to Canvas. The canvas deals with the computation of the vertices.

        """
        # Failure cases?
        with self.lock:
            self.update_panes(target)
            self.targets[target] = layers
        return True

    def set_model_struct(self, struct):
        """
            This method should actually be able to infer the
            pane + reshaping parameters from a model struct
        """
        #self.struct = struct
        pass

    def get_processable_targets(self, data):
        """
        """
        ret = []
        for target in self.targets:
            if all(layer in data.keys() for layer in self.targets[target]):
                ret.append(target)
        return ret

    def process_base_layer(self, data, target):
        """
            DOC
        """
        ### RESHAPE
        layer_id    = target[0]
        data        = data[layer_id]
        n_item      = data.shape[1]
        new_shape   = [n_item] + list(reshape_table[layer_id])
        new_texture = data.transpose().astype(np.float32)
        new_texture = new_texture.reshape(new_shape)
        new_texture = resize_nearest(new_texture, reshape_table[0])

        ### UPDATE TEXTURES
        tex_range = self.panes.get_texture_id(target)
        self._log_shapes()
        self.textures[tex_range,:,:] = new_texture
        log.debug(
            'Layer {} texture: shape={}, valeur max={}, valeur min={}'.format(layer_id,
                                                                              new_texture.shape,
                                                                              np.max(new_texture),
                                                                              np.min(new_texture))
                  )
        log.debug('max(data)={}'.format(np.max(new_texture)))
        log.debug('min(data)={}'.format(np.min(new_texture)))
        return True

    def process_target(self, data, target):
        """
            Target structure is ['solo/cumul', layerName, nodeId]
        """
        if target[0] in self.struct:
            if target[1] == None:
                if target[2] != -1:
                    #TODO: Demander à ne pas visualiser tous les poids de la layer mais seulement certains
                    log.error('Not yet implemented node only display target: {}'.format(target))
                    #TODO: return False. For test purpose we still process it as normal
                    return self.process_base_layer(data, target)
                else:
                    # TODO: Handle different kind of layer
                    log.debug('processing target {}'.format(target))
                    return self.process_base_layer(data, target)
            else:
                # TODO: Cumul des poids
                log.error('Not yet implemented cumul display target {}'.format(target))
                return False
        else:
            log.error('Layer not available. Asked for target {} but struct is {}'.format(target, self.struct))
            return False

    def process_data(self, data):
        """
            Process data received from the queue.
        """
        with self.lock:
            processable_targets = self.get_processable_targets(data)
            log.debug('Targets {} are processable with received data {}'.format(processable_targets, data.keys()))
            for target in processable_targets:
                processed_textures = self.process_target(data, target)
